from SoccerNet.Downloader import SoccerNetDownloader as SNdl

local_directory = "./data"
mySNdl = SNdl(LocalDirectory=local_directory)
mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])