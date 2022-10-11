# Cat &amp; dog binary classification with PyTorch

### Accuracy on test set: 98.2667%
<br>
Used dataset: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset
<br>
<br>
Time of training (with validation phases) 4.0h 15.0m 25s.
<br>
<br>

<br>
<img src= "utils/architecture.png">
<br>
<img src= "utils/plot_acc.png" width=600 height=450>
<br>
<img src= "utils/plot_loss.png" width=600 height=450>
<br>
<img src= "utils/plot_lr.png">
<br>
<img src= "utils/plot_cm.png">
Confusion matrix was created from the predictions on the TEST set
<br>
<br>


Total parameters:   19,913,346<br>
Total trainable parameters:   19,913,346<br>
<br>
Training dataset size: 		17,998<br>
Validation dataset size: 	4,000<br>
Test dataset size: 		3,000<br>
<br>
Input image size: 256 × 256<br>
<br>
Batch size:    	 32<br>
Learning rate: 	 0.0005<br>
Loss function: 	 cross_entropy<br>
No. epochs:    	 50<br>
<br>


Epoch: 1<br>
TRAINING     train accuracy 60.28%, train loss: 0.6706<br>
VALIDATION   val accuracy   67.05%, val loss:   0.6114<br>
<br>
Epoch: 2<br>
TRAINING     train accuracy 65.92%, train loss: 0.6149<br>
VALIDATION   val accuracy   69.30%, val loss:   0.5610<br>
<br>
Epoch: 3<br>
TRAINING     train accuracy 67.89%, train loss: 0.5890<br>
VALIDATION   val accuracy   59.88%, val loss:   0.7005<br>
<br>
Epoch: 4<br>
TRAINING     train accuracy 69.77%, train loss: 0.5689<br>
VALIDATION   val accuracy   74.62%, val loss:   0.5101<br>
<br>
Epoch: 5<br>
TRAINING     train accuracy 72.88%, train loss: 0.5301<br>
VALIDATION   val accuracy   79.33%, val loss:   0.4366<br>
<br>
Epoch: 6<br>
TRAINING     train accuracy 76.88%, train loss: 0.4768<br>
VALIDATION   val accuracy   84.38%, val loss:   0.3576<br>
<br>
Epoch: 7<br>
TRAINING     train accuracy 79.23%, train loss: 0.4369<br>
VALIDATION   val accuracy   88.05%, val loss:   0.2972<br>
<br>
Epoch: 8<br>
TRAINING     train accuracy 81.59%, train loss: 0.3945<br>
VALIDATION   val accuracy   88.28%, val loss:   0.2804<br>
<br>
Epoch: 9<br>
TRAINING     train accuracy 84.13%, train loss: 0.3553<br>
VALIDATION   val accuracy   86.58%, val loss:   0.3031<br>
<br>
Epoch: 10<br>
TRAINING     train accuracy 85.09%, train loss: 0.3317<br>
VALIDATION   val accuracy   92.30%, val loss:   0.2051<br>
<br>
Epoch: 11<br>
TRAINING     train accuracy 86.85%, train loss: 0.2989<br>
VALIDATION   val accuracy   92.83%, val loss:   0.1906<br>
<br>
Epoch: 12<br>
TRAINING     train accuracy 87.00%, train loss: 0.2944<br>
VALIDATION   val accuracy   92.53%, val loss:   0.1889<br>
<br>
Epoch: 13<br>
TRAINING     train accuracy 87.60%, train loss: 0.2817<br>
VALIDATION   val accuracy   92.38%, val loss:   0.2007<br>
<br>
Epoch: 14<br>
TRAINING     train accuracy 88.25%, train loss: 0.2667<br>
VALIDATION   val accuracy   93.33%, val loss:   0.1615<br>
<br>
Epoch: 15<br>
TRAINING     train accuracy 88.88%, train loss: 0.2529<br>
VALIDATION   val accuracy   91.95%, val loss:   0.2161<br>
<br>
Epoch: 16<br>
TRAINING     train accuracy 89.74%, train loss: 0.2399<br>
VALIDATION   val accuracy   93.45%, val loss:   0.1630<br>
<br>
Epoch: 17<br>
TRAINING     train accuracy 90.50%, train loss: 0.2267<br>
VALIDATION   val accuracy   94.38%, val loss:   0.1306<br>
<br>
Epoch: 18<br>
TRAINING     train accuracy 91.27%, train loss: 0.2146<br>
VALIDATION   val accuracy   92.45%, val loss:   0.2027<br>
<br>
Epoch: 19<br>
TRAINING     train accuracy 90.79%, train loss: 0.2184<br>
VALIDATION   val accuracy   94.85%, val loss:   0.1390<br>
<br>
Epoch: 20<br>
TRAINING     train accuracy 91.44%, train loss: 0.2081<br>
VALIDATION   val accuracy   93.22%, val loss:   0.1702<br>
<br>
Epoch: 21<br>
TRAINING     train accuracy 91.87%, train loss: 0.1939<br>
VALIDATION   val accuracy   94.10%, val loss:   0.1640<br>
<br>
Epoch: 22<br>
TRAINING     train accuracy 92.19%, train loss: 0.1897<br>
VALIDATION   val accuracy   95.72%, val loss:   0.1066<br>
<br>
Epoch: 23<br>
TRAINING     train accuracy 93.11%, train loss: 0.1669<br>
VALIDATION   val accuracy   95.15%, val loss:   0.1219<br>
<br>
Epoch: 24<br>
TRAINING     train accuracy 92.67%, train loss: 0.1759<br>
VALIDATION   val accuracy   95.55%, val loss:   0.1205<br>
<br>
Epoch: 25<br>
TRAINING     train accuracy 93.42%, train loss: 0.1616<br>
VALIDATION   val accuracy   96.15%, val loss:   0.0979<br>
<br>
Epoch: 26<br>
TRAINING     train accuracy 93.80%, train loss: 0.1520<br>
VALIDATION   val accuracy   96.12%, val loss:   0.1035<br>
<br>
Epoch: 27<br>
TRAINING     train accuracy 94.24%, train loss: 0.1468<br>
VALIDATION   val accuracy   96.33%, val loss:   0.0991<br>
<br>
Epoch: 28<br>
TRAINING     train accuracy 94.51%, train loss: 0.1346<br>
VALIDATION   val accuracy   96.50%, val loss:   0.0891<br>
<br>
Epoch: 29<br>
TRAINING     train accuracy 94.58%, train loss: 0.1343<br>
VALIDATION   val accuracy   95.62%, val loss:   0.1032<br>
<br>
Epoch: 30<br>
TRAINING     train accuracy 94.78%, train loss: 0.1277<br>
VALIDATION   val accuracy   96.92%, val loss:   0.0828<br>
<br>
Epoch: 31<br>
TRAINING     train accuracy 94.98%, train loss: 0.1260<br>
VALIDATION   val accuracy   95.90%, val loss:   0.0934<br>
<br>
Epoch: 32<br>
TRAINING     train accuracy 95.48%, train loss: 0.1131<br>
VALIDATION   val accuracy   96.97%, val loss:   0.0899<br>
<br>
Epoch: 33<br>
TRAINING     train accuracy 95.51%, train loss: 0.1114<br>
VALIDATION   val accuracy   96.88%, val loss:   0.0820<br>
<br>
Epoch: 34<br>
TRAINING     train accuracy 95.87%, train loss: 0.1038<br>
VALIDATION   val accuracy   96.97%, val loss:   0.0841<br>
<br>
Epoch: 35<br>
TRAINING     train accuracy 95.96%, train loss: 0.1048<br>
VALIDATION   val accuracy   97.20%, val loss:   0.0764<br>
<br>
Epoch: 36<br>
TRAINING     train accuracy 96.27%, train loss: 0.0941<br>
VALIDATION   val accuracy   97.12%, val loss:   0.0724<br>
<br>
Epoch: 37<br>
TRAINING     train accuracy 96.18%, train loss: 0.0926<br>
VALIDATION   val accuracy   96.88%, val loss:   0.0820<br>
<br>
Epoch: 38<br>
TRAINING     train accuracy 96.51%, train loss: 0.0851<br>
VALIDATION   val accuracy   97.28%, val loss:   0.0727<br>
<br>
Epoch: 39<br>
TRAINING     train accuracy 96.78%, train loss: 0.0818<br>
VALIDATION   val accuracy   97.22%, val loss:   0.0766<br>
<br>
Epoch: 40<br>
TRAINING     train accuracy 96.94%, train loss: 0.0767<br>
VALIDATION   val accuracy   97.33%, val loss:   0.0725<br>
<br>
Epoch: 41<br>
TRAINING     train accuracy 97.04%, train loss: 0.0772<br>
VALIDATION   val accuracy   97.28%, val loss:   0.0753<br>
<br>
Epoch: 42<br>
TRAINING     train accuracy 97.10%, train loss: 0.0710<br>
VALIDATION   val accuracy   97.50%, val loss:   0.0706<br>
<br>
Epoch: 43<br>
TRAINING     train accuracy 97.16%, train loss: 0.0678<br>
VALIDATION   val accuracy   97.65%, val loss:   0.0705<br>
<br>
Epoch: 44<br>
TRAINING     train accuracy 97.62%, train loss: 0.0605<br>
VALIDATION   val accuracy   97.42%, val loss:   0.0712<br>
<br>
Epoch: 45<br>
TRAINING     train accuracy 97.33%, train loss: 0.0653<br>
VALIDATION   val accuracy   97.55%, val loss:   0.0695<br>
<br>
Epoch: 46<br>
TRAINING     train accuracy 97.68%, train loss: 0.0601<br>
VALIDATION   val accuracy   97.42%, val loss:   0.0726<br>
<br>
Epoch: 47<br>
TRAINING     train accuracy 97.47%, train loss: 0.0648<br>
VALIDATION   val accuracy   97.45%, val loss:   0.0710<br>
<br>
Epoch: 48<br>
TRAINING     train accuracy 97.32%, train loss: 0.0635<br>
VALIDATION   val accuracy   97.50%, val loss:   0.0708<br>
<br>
Epoch: 49<br>
TRAINING     train accuracy 97.52%, train loss: 0.0620<br>
VALIDATION   val accuracy   97.50%, val loss:   0.0699<br>
<br>
Epoch: 50<br>
TRAINING     train accuracy 97.71%, train loss: 0.0585<br>
VALIDATION   val accuracy   97.53%, val loss:   0.0702<br>
<br>
<br>

TIME of training (with validation phases) 4.0h 15.0m 25s.
