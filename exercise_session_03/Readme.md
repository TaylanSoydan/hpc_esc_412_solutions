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

Below is how PCS scaled with different threads on my local. In theory, PCS should have the worst scaling as calculating 64 weights and updating 64 cells becomes the overhead in computation where CIC should have the best scaling for the same reason. 

![image](https://user-images.githubusercontent.com/73917265/225910688-5e3865b5-ff5f-4a23-a0c4-683cdbff24d3.png)

