sudo docker stop $( sudo docker ps -aq | awk '{print $1}')
sudo docker rm $( sudo docker ps -aq | awk '{print $1}')
