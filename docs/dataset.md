
# Downloading the VSC2022 challenge data 

The list of video and ground-truth files is here: [vsc_url_list.txt](https://dl.fbaipublicfiles.com/video_similarity_challenge/46ef53734a4/vsc_url_list.txt)

The video files can be downloaded in a single `wget` command: 
```
wget -i https://dl.fbaipublicfiles.com/video_similarity_challenge/46ef53734a4/vsc_url_list.txt \
  --cut-dirs 2 -x -nH
```

