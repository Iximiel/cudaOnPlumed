# Rewriting the coordination for plumed in Cuda

Here I am showing how to set up a plug-in that is compiled with the cuda compiler and that can be LOADed in plumed.

The project consist in two part: the actual [coordination implementation](Implementation.md) and an [helper module](Helpers.md) with the reduction algorithm and a tool for ease the memory management.


```mermaid
flowchart LR

Intro[Introduction]
Implementation
Helpers

Intro ==> Implementation
Implementation <==> Helpers
Intro ==> Helpers


click Intro "Navigation.md" "The introduction"
click Implementation "Implementation.md" "The coordination implementation"
click Helpers "Helpers.md" "A simple manual witht the helper library"
```
