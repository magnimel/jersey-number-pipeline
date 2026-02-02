from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from os import rename

local_directory = "./data"
mySNdl = SNdl(LocalDirectory=local_directory)
mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])
rename(f"{local_directory}/jersey-2023", f"{local_directory}/SoccerNet")
