# LLM Training Log

## 04/18

I plan on training a ~850m parameter llm from scratch using google colab and kaggle's free GPU hours. I plan to add updates as I got here regarding loss, states, and any other updates that may come. 
I'm currently running a smoke test then I will move on to a pilot test. The pilot test will be 11.5 hours on kaggle on a 150m parameter model training on 0.5b tokens. (I may run two depending on results for a total of 1b tokens)

## 04/19

The pilot test went super well reach a loss of roughly 3.5. I noticed some weird losses on specific datasets (0.2, 1.0) which I realized were datasets that repeated numbers over and over again. like how many zeros are in said number 00000000000000000... so I updated gen_dataset, regenerated the dataset and I'm starting a second pilot test, which should be the last test before we commence the full training run for the 800m parameter model. 

## 04/20

I realized today that I had some AWS credits so I'm going to start training the big model on an AWS g6e which hopefully should give me a major jump start on this. Then I'll head back to kaggle and google colab to finish the next 9mo of training. (I'm currently predicting it will take less than that because training has been going faster than I thought). I also finished pulling the dataset and converting it for the big run.
