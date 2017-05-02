# Results for pix2pix and first CycleGAN

|name|input| pix2pix| CycleGAN (u-net)| CycleGAN (faststyle-net)|target|
|---|---|---|---|---|---|
|1|![](run_1_images/1-inputs.png)|![](run_1_images/1-outputs-pix2pix.png)|![](run_1_images/1-outputs-CycleGAN.png)| ![](run_1_images/1-outputs-CycleGAN-faststyle.png) |![](run_1_images/1-targets.png)|

Notes:

*   These examples are taken during the training (~50 epochs).
*   CycleGAN uses a u-net or faststyle-net generators, log loss and a batch-size of 1
*   Note: CycleGAN has a nice output, but does not map windows labels to windows patterns 

|pix2pix|CycleGAN|
|---|---|
|![](run_1_images/Graph_Pix2Pix.png)|![](run_1_images/Graph_CycleGAN.png)|

|u-net|faststyle_net|
|---|---|
|![](run_1_images/Graph_u_net.png)|![](run_1_images/Graph_faststyle_net.png)|
