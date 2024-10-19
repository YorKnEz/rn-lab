## Results

learn_rate | epochs  | accuracy   | time
-----------|---------|------------|-----------
0.01       | 20      | 0.923      | ~3m 15s   
0.01       | 20      | 0.9227     | ~3m 15s   
0.01       | 100     | 0.9103     | ~15m      
0.001      | 20      |  0.9234    | ~3m 15s   
0.001      | 20      |  0.9235    | ~3m 15s   
0.001      | 20      | 0.9243     | 1m 0s     
0.001      | 20      | 0.9232     | 1m 0s     
0.001      | 50      | 0.9247     | 2m 33s    
0.001      | 100     | 0.925      | 15m 26s   
**0.001**  | **100** | **0.9269** | **5m 3s** 
0.0001     | 20      | 0.9121     | ~3m 15s   
0.0001     | 20      | 0.9117     | ~3m 15s   
**0.0001** | **100** | **0.9269** | **5m 8s** 
0.0001     | 500     | 0.9257     | 26m 30s   

## Notes

`0.001` seems to be the best `learn_rate` from these few experiments
`100` epochs seems to yield the best accuracy

The `train` function used has been optimized between tests, lowering the total time by a factor of 3 (that is why, for example, running the algorithm with `learning_rate = 0.001, epochs = 100` takes both 15m and 5m).