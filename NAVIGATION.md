# Rewriting the coordination for plumed in Cuda

Here I am showing how to set up a plug-in that is compiled with the cuda compiler
and that can be LOADed in plumed.
The project consists in two parts: the actual
[coordination implementation](Implementation.md) and a [helper module](Helpers.md)
 with the reduction algorithm and a tool for easing memory management.


```mermaid
flowchart LR

Intro[Introduction]
Implementation
Helpers
AB[GROUPA,GROUPB]
Pair

Intro ==> Implementation
Implementation <==> Helpers
Intro ==> Helpers
Implementation <==> Pair
Implementation <==> AB


click Intro "NAVIGATION.md" "The introduction"
click Implementation "Implementation.md" "The coordination implementation"
click AB "ImplementationTwoGroups.md" "The coordination between two groups"
click Pair "ImplementationPair.md" "The coordination in pairs"
click Helpers "Helpers.md" "A simple manual with the helper library"
```
