
# coding: utf-8

# In[ ]:


import socket  
import base64
import cv2

HOST = 'localhost'  
PORT = 8888
BUFSIZ = 102400  
ADDR = (HOST, PORT)  
  
tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
tcpCliSock.connect(ADDR)  
  



cap = cv2.VideoCapture(0)
j=0
while(cap.isOpened()):
    ret, frame = cap.read()
    j=j+1
    
        
    if ret == True:
	cv2.imshow("capture", frame)
	if(j%1==0):
	    cv2.imwrite("/home/forest/图片/2.jpg", frame)
	    f=open(r'/home/forest/图片/2.jpg','rb') #二进制方式打开图文件
	    data=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
	    f.close()
	    data=data+"@@END@@"
	    tcpCliSock.send("%s"%data) 
	    data2 = tcpCliSock.recv(BUFSIZ)  
	    print data2.strip()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


# In[66]:


