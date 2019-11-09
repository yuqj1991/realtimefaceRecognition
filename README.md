# real_time_object_tracking
***
*@change log: initial by Xiaoguang Zhao* *8/27/2019*
***
## prepare for git environmnet 
**config user name and email address**  
   > git config --global user.name "`your name`"  
   > git config --global user.email "`your resideo email`"  

####
# **Features**
## 主要实现的功能
是实现通过深度学习，检测物体（人脸），之后再实现跟踪算法
从流程来看，需要实行
1. 调用加载模型文件， 
2. 加载跟踪算法，
3. 然后就会有个问题  
从模块来说，
1. 物体检测模块；
2. 人脸跟踪模块；
3. 错误代码提示模块；
4. 总工程提示模块（人脸检测一帧，测试)
    *其实就是比较困惑的是要是再有人脸进来该如何处理呢，每次间隔5-10帧，目前先这样处理*  
facedetect: 输入图片大小320, 320
mean_: 103.94, 116.78, 123.68
std: 0.007843
faceangle: 输入96,96
mean_: 127.5, 127.5, 127.5
std: 0.007845
faceattri: 输入 输入96,96
mean_: 127.5, 127.5, 127.5
std: 0.007845
facenet 160, 160
mean_: 127.5, 127.5, 127.5
std: 0.007845


## *how to checkout to your local folder*
use `git clone` command to checkout code as below, the repo address is `https://bitbucket.honeywell.com/scm/aiproject/facerec.git`  

so type command as below in your proper folder:

> `git clone https://bitbucket.honeywell.com/scm/aiproject/facerec.git`  

then you are asked to input username for bitbucket, here is your `EID`, the password is your `LDAP password`:
for example, the EID is `e361479` 
> Username for 'https://bitbucket.honeywell.com': `e361479`  
> Password for 'https://e361479@bitbucket.honeywell.com': `password`  

If all these information is correct, git will start to pull repo to local. 

For further commit your changes, please refer to git documents. 

