Ex1:

Program outputs phase times now as such:

![image](https://user-images.githubusercontent.com/73917265/225765151-4f4041de-8441-44cc-bfa9-d1b0459f79f5.png)

Ex2:

I tried the 100, 200, 500 files and got these durations.

![image](https://user-images.githubusercontent.com/73917265/225765223-88715cca-0433-4ef5-b035-d0478b97a2af.png)

Assignment function should be linear O(N) as we have a for loop that iterates over N.

Projection depends on the grid dimensions, should be O(i*j*k) as we have 3 for loops iterating over dimensions.

Ex3:

results and codes are included for each method in its own folder.

Ex4:

For example on the trial of CIC with OMP_NUM_THREADS=4 vs 1, the assignment got nearly 20% faster. PCS had the worst scaling as calculating 64 weights and updating 64 cells becomes the overhead in computation where CIC had the best scaling for the same reason. 

CIC - serial
Mass assignment took: 902.783 seconds

CIC - parallel
Mass assignment took: 733.118 seconds
