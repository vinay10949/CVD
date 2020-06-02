sudo docker stop $( sudo docker ps -aq | awk '{print $1}')
sudo docker rm $( sudo docker ps -aq | awk '{print $1}')
sudo docker rmi $(sudo docker images | grep '<none>' | awk '{print $3}')
sudo docker-compose -f docker-compose.yml up -d --build
