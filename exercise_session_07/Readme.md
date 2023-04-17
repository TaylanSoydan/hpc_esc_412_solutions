Ex 1:

I added print (blitz (sum( array))) statements to the codes from exercise 03 and run them. The total masses do not add up to 10^6. 

![Ekran görüntüsü 2023-03-27 121842](https://user-images.githubusercontent.com/73917265/228578190-201b7d36-fcc5-4cf4-9aa6-c3c5643ef649.png)

Ex 2:

I modified the weights file and tested it and got (i added print(nBAD) to be sure) 

root@DESKTOP-4755H6G:~/hpc_esc_412_solutions/exercise_session_05/ex05# ./test_weights

nBAD = 0



Ex 3:

I created an assign.cxx script to run all methods one by one and check the sums, this time they added up to 10^6.

root@DESKTOP-4755H6G:~/hpc_esc_412_solutions/exercise_session_05/ex05# ./assign ../../exercise_session_04/ex02/B100.00100 100

Loading 1000000 particles

ngp total = 1e+06

cic total = 1e+06

tsc total = 1e+06

pcs total = 1e+06

Reading file took: 0.0360659 seconds

Mass assignment took: 0.1346 seconds

Projection took: 0.239509 seconds
