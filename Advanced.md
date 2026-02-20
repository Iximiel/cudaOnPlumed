# Advanced implementation

I am assuming that you have at least read the [base version](Implementation.md) of the implementation procedure.

Here I will introduce a few much more advanced concepts:

 - the link cells algorithm, as described by Allen, and as implemented in plumed in the LinkCells class (you can get it with `"#include "plumed/tools/LinkCells.h"`)
 - What is the shared memory in Cuda

### How does LinkCells work?

The `LinkCells` algorithm is similar to the `NeighborList` one, but the list of neighbors will be shared by a group of atoms.
In a few words: every _N_ steps, typically `N==1`, the following algorithm is run:

 - Divide the simulation box into cells
 - Assign each atom to the cell in which it is contained
 - Determine the list of “neighbor” cells for each cell (the adjacent ones, also in diagonal, for example, in the image below, for `6` are `1 2 3 5 7 9 10 11`), in 3D, the neighbor cells will be 26 -or fewer if the box is sliced into fewer boxes-
 - Assign to each cell the _candidate neighbors_ atoms by merging the list of atoms of the central cell and the ones of the neighbors cells.

For example in the image below all the atoms in cell `6` share the list of _candidate neighbors_ that are the atoms contained in the e `1 2 3 5 6 7 9 10 11` cells.

In this way for collection of atoms assigned to a specific cell we will have a shared neighbors list.
Thil will be used to make the GPU process less data and to save some memory.

![](./grid.svg)

### Shared memory in Cuda

The shared memory in Cuda is a (small) portion of memory that is shared between the threads in a group of a kernel. It acts as a sort of level3 cache for those kernels, so that accessing the data in the shared memory is significantly faster than accessing the data in the RAM of the GPU.

Moreover, you can build the shared memory in order to access it in a less sparse way than you would access the standard RAM. This “massage” of the memory is a small tradeoff and usually ends up giving a huge advantage in speed.

## Setting up the algorithm

Now I will explain a simplified version of the CudaCoordination implementation that will be shipped with PLUMED v2.11.

We want to do a few things:

 - On the CPU:
  - We calculate the linkCells
  - We assign the addresses of the atoms to the various cells
  - We create the _candidate neighbors_ list for each cell and we store it on the CPU
  - We transfer all the data from CPU to GPU
  - We instruct the GPU on the dimensions of the cells an of the _candidate neighbors_ lists

 - On the GPU:
  - We load the shared memory
  - We calculate the Coordination, "as usual"
 - Back on the CPU:
  - We collect and we reduce coordination and virial, as we did in the Implementation example.

We declare the classes that does for us the LinkCell algorithm:
```c++
LinkCells cells;
LinkCells::CellCollection listA;
```
We  declare a companion class:

```c++
class cellSetup {
  struct lims {
    unsigned start=0;
    unsigned size=0;
  };
  lims atomsInCells;
  lims otherAtoms;
  lims nat_InCells;
  lims nat_otherAtoms;
  std::vector<unsigned> data{};
  unsigned maxNeighbors=0;
  unsigned maxAtomsPerCell=0;
public:
  unsigned maxExpected()const {
    return maxNeighbors;
  }
  unsigned biggestCell()const {
    return maxAtomsPerCell;
  }
  View<const unsigned> dataView() const {
    return make_const_view(data);
  }
  void reset(
    const unsigned ncells,
    const unsigned biggest_cell, //the number of atoms in the biggest cell
    const unsigned cellUpper,// the error value for the cell
    const unsigned max_expected,// the maximum number of atoms in the neibourhood
    const unsigned otherUpper//the error value for the neighbourood
  ) {
    maxNeighbors=max_expected;
    maxAtomsPerCell=biggest_cell;
    //this resizes and set to zero the part of the array that stores the number
    // of atoms in cell or in the neigbor list for the cells
    data.assign((maxAtomsPerCell+maxNeighbors+1+1)*ncells,0);
    atomsInCells= {0,ncells*maxAtomsPerCell};
    otherAtoms= {ncells*maxAtomsPerCell,maxNeighbors*ncells};
    nat_InCells= {(maxAtomsPerCell+maxNeighbors)*ncells,ncells};
    nat_otherAtoms= {(maxAtomsPerCell+maxNeighbors+1)*ncells,ncells};
    //we test in production
    plumed_assert(data.size() ==
                  get_atomsInCells().size()
                  +get_otherAtoms().size()
                  +get_nat_InCells().size()
                  +get_nat_otherAtoms().size()) << "the view in cellSetup are not built correcly";

    auto tmp_nat_otherAtoms = get_nat_otherAtoms();
    plumed_assert(&data[data.size()-1] == &tmp_nat_otherAtoms[tmp_nat_otherAtoms.size()-1])
        << "the view may not be correctly set up";
    auto aic=get_atomsInCells();
    std::fill(aic.begin(),aic.end(),cellUpper);
    auto oa=get_otherAtoms();
    std::fill(oa.begin(),oa.end(),otherUpper);
  }
  PLMD::View<unsigned> get_atomsInCells() {return data.data()+atomsInCells.start,atomsInCells.size};}
  PLMD::View<unsigned> get_otherAtoms() {return data.data()+otherAtoms.start,otherAtoms.size};}
  PLMD::View<unsigned> get_nat_InCells() {return data.data()+nat_InCells.start,nat_InCells.size};}
  PLMD::View<unsigned> get_nat_otherAtoms() {return data.data()+nat_otherAtoms.start,nat_otherAtoms.size};}
    return thrust::raw_pointer_cast (deviceData.data()) + tp.start; }
  unsigned* deviceMap_atomsInCells(thrust::device_vector<unsigned> deviceMap(atomsInCells)deviceData) const {
    return thrust::raw_pointer_cast (deviceData.data()) + atomsInCells.start; }
  unsigned* deviceMap_otherAtoms(thrust::device_vector<unsigned> deviceMap(otherAtoms)deviceData) const {
    return thrust::raw_pointer_cast (deviceData.data()) + otherAtoms.start; }
  unsigned* deviceMap_nat_InCells(thrust::device_vector<unsigned> deviceMap(nat_InCells)deviceData) const {
    return thrust::raw_pointer_cast (deviceData.data()) + nat_InCells.start; }
  unsigned* deviceMap_nat_otherAtoms(thrust::device_vector<unsigned> deviceMap(nat_otherAtoms)deviceData) const {
    return thrust::raw_pointer_cast (deviceData.data()) + nat_otherAtoms.start; }
} cellConfiguration_coord;
```
this `cellSetup` is a trick class to pack all the informations in a single array, to reduce the number of data passages between CPU and GPU

Below you will see a NL variable in a few places, it is defined like this:
```c++
struct nlSettings {
  calculateFloat cutoff=-1.0;
  int stride=-1;
} NL;
```

#### Start with setting up the _candidate neighbors_
Note that here I removed all the MPI safeguard for clarity.`LinkCells` must run on all the MPI processes, but the calculations usually needs to run ONLY on a limited subset (1) of GPUs, so it is possible to run the "memory preparation" instructions only on the main process.

```c++
cells.setCutoff(NL.cutoff);
if (pbc) {
  cells.setupCells(getPbc());
} else {
  cells.setupCells(getPositions());
}
//LinkCells needs the list of the indexes of the atoms
std::vector<unsigned> indexesForCells(getPositions().size());
std::iota(indexesForCells.begin(),indexesForCells.end(),0);
const size_t nat = cudaPositions.size() / 3;
//this needs to be done BY ALL the mpi processes
cells.resetCollection(listA,
                      make_const_view(getPositions()),
                      make_const_view(indexesForCells));
//It is not necessary to run this on mpi processes that do not access to the GPU
//if (mpiActive){

const auto maxExpected = listA.getMaximimumCombination(27);
const auto biggestCell = (*std::max_element(listA.lcell_tots.begin(),cs.listA.lcell_tots.end()));
//Asserts make clear to who read the code what is the "contract" of the function
//In this case there is an hard limit on the number of atoms in the inner cell
//This number is around 600 atoms in the PLUMED implementation
plumed_assert(biggestCell <= maxNumThreads)
    << "the number of atoms in a single cell("<<biggestCell
    <<") exceds the maximum number of threads avaiable("<<maxNumThreads
    <<"), try by reducing the cell dimensions or by increase the THREADS asked";

cellConfiguration_coord.reset(cells.getNumberOfCells(),
                              biggestCell,
                              nat,
                              maxExpected,
                              nat);

updateCellists(cells,
               listA,
               listA,//used again because we have only a single list of atoms in this case
               nat,
               maxExpected,
               biggestCell,
               pbc,
               cellConfiguration_coord.get_atomsInCells(),
               cellConfiguration_coord.get_otherAtoms(),
               cellConfiguration_coord.get_nat_InCells(),
               cellConfiguration_coord.get_nat_otherAtoms());
```

`updateCellist` has the following shape:
```c++
void updateCellists(
  const PLMD::LinkCells& cells,
  const PLMD::LinkCells::CellCollection& listCell,
  const PLMD::LinkCells::CellCollection& listNeigh,
  const unsigned upperCheck,// for checking
  const unsigned maxNNExpected,
  const unsigned biggestCell,
  const bool pbc,
  PLMD::View<unsigned> atomsIncells,
  PLMD::View<unsigned> otherAtoms,
  PLMD::View<unsigned> nat_InCells,
  PLMD::View<unsigned> nat_otherAtoms
) {
  // we are going to extract the data from the CellCollections into the cellSetup
  //the nat_* lists store how may atoms are in that segment of the cell
  //the otherAtoms and atomsIncells will be organized like this:
  //[c_0id0,c_0id1,..., // starting at position [0]
  // c_nid0,c_nid1,...  // starting at position [n*(biggestCell or maxNNExpected)]
  // So we oly need to store the dimension of the sublist and not the starting position, that can be calculated when needed
  std::vector<unsigned> cells_required(27);
  for(unsigned c =0; c < cells.getNumberOfCells() ; ++c) {
    auto atomsInC= listCell.getCellIndexes(c);
    if(atomsInC.size()>0) {
      plumed_assert(atomsInC.size() <= biggestCell);
      nat_InCells[c]=atomsInC.size();
      auto cell = cells.findMyCell(c);
      unsigned ncells_required=0;
      cells.addRequiredCells(cell,ncells_required, cells_required,pbc);
      //we copy the idx of the inner cell in the InCells list
      std::copy(atomsInC.begin(),atomsInC.end(),atomsIncells.begin()+biggestCell*c);
      unsigned otherAtomsId=c*maxNNExpected;
      //we copy the idx of the candidate atoms in the otherAtomList
      for (unsigned cb=0; cb <ncells_required ; ++cb) {
        for (auto B : listNeigh.getCellIndexes(cells_required[cb])) {
          otherAtoms[otherAtomsId]=B;
          ++otherAtomsId;
        }
      }
      nat_otherAtoms[c]=otherAtomsId - c*maxNNExpected;
      if(nat_otherAtoms[c]>0) {
        //no sense in risking to check the address before the start of otherAtoms or to check if no atoms have been added
        plumed_assert(otherAtoms[otherAtomsId-1]<upperCheck);
      }
      plumed_assert(nat_otherAtoms[c] <= maxNNExpected);
      plumed_assert(nat_InCells[c] <= biggestCell);
    }
  }
}
```

Then we finally move the atomic data on the GPU:

