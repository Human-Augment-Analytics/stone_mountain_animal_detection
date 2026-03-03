#Date and Time Management
#Initializations----------------------
#load package necessary for date corrections
library(lubridate) #for conversion
library(exifr) #for extraction

setwd("F:/Camera Trap Photos") #set to working directory of flash-drive/data repository
#Load observation data
DF <- read.csv("Observations_Nov21.csv")

#Extraction---------------------------
#Define batch
Year <- 2022
Batch <- "Nov 21"

NumLoc <- unique(DF$Location)

for(l in 1:length(NumLoc)) {
  FocLoc <- NumLoc[l]
  FilePath <- file.path(getwd(), FocLoc, Year, Batch)
  
  FocDF <- DF[which(DF$Location == FocLoc), ]
  for(e in 1:nrow(FocDF)) {
    if(FocDF$Species[e] == "") {
      next()
    }
    
    FocusJPG <- file.path(FilePath, paste0(FocDF$ImageName[e], ".jpg"))
    
    Metadata <- exifr::read_exif(FocusJPG)
    metadate <- Metadata$DateTimeOriginal
  
    Test <- lubridate::ymd_hms(metadate)
    
    metatime <- strsplit(as.character(Test), " ")[[1]][2]
    
    FocDF$CameraDate[e] <- as.character(lubridate::date(Test))
    FocDF$CameraTime[e] <- metatime
    print(e)
  }
  
  if(l == 1) {
    NewDF <- FocDF
  } else {
    NewDF <- rbind(NewDF, FocDF)
  }
  message(l)
}



#Conversion---------------------------
#Convert Extraction Date (CameraDate) to actual data (TrueDate)
#Extract Date and Time columns
Date <- DF$CameraDate
Time <- DF$CameraTime

#Paste them together
DateTime <- rep(NA, length = length(Date))

for(d in 1:length(DateTime)) {
  DateTime[d] <- paste0(Date[d], " ", Time[d])
}

#Convert to lubridate format
DateTime2 <- mdy_hm(DateTime)

#TODO Update after finishing all photos up to July 27, 2022
#Note: the offsets may be different as each camera trap's date/time was updated  
Offset_SM1 <- 0.05833333
Offset_SM2 <- 1866.502083
Offset_SM3 <- 1866.606944
Offset_SM4 <- 1866.636111
Offset_SM5 <- 1866.634028

DF$TrueDate <- rep(NA, length = nrow(DF))
DF$TrueTime <- rep(NA, length = nrow(DF))

for(e in 1:nrow(DF)) {
  if(DF$Location[e] == "SM_1") {
    
    #Add offset (for SM1, the offset is always added)
    TrueDateTime <- DF$CameraTime[e] + duration(Offset_SM1, units = "days")
    #Round date to nearest minute and convert to character strings of date and time
    TrueDateTime <- round_date(TrueDateTime, unit = "minute")
    TrueDateTime <- strsplit(as.character(TrueDateTime), split = " ")[[1]]
    
    DF$TrueDate[e] <- TrueDateTime[1]
    DF$TrueTime[e] <- TrueDateTime[2]
  } else if (DF$Location[e] == "SM_2") {
    
    #If the camera trap date says 2017, add the offset
    if (year(DateTime2[e]) == "2017") {
      TrueDateTime <- DateTime2[e] + duration(Offset_SM2, units = "days")
    } else {
      TrueDateTime <- DateTime2[e]
    }
    #Round date to nearest minute and convert to character strings of date and time
    TrueDateTime <- round_date(TrueDateTime, unit = "minute")
    TrueDateTime <- strsplit(as.character(TrueDateTime), split = " ")[[1]]
    
    DF$TrueDate[e] <- TrueDateTime[1]
    DF$TrueTime[e] <- TrueDateTime[2]
  } else if (DF$Location[e] == "SM_3") {
    
    #If the camera trap date says 2017, add the offset
    if (year(DateTime2[e]) == "2017") {
      TrueDateTime <- DateTime2[e] + duration(Offset_SM3, units = "days")
    } else {
      TrueDateTime <- DateTime2[e]
    }
    #Round date to nearest minute and convert to character strings of date and time
    TrueDateTime <- round_date(TrueDateTime, unit = "minute")
    TrueDateTime <- strsplit(as.character(TrueDateTime), split = " ")[[1]]
    
    DF$TrueDate[e] <- TrueDateTime[1]
    DF$TrueTime[e] <- TrueDateTime[2]
  } else if (DF$Location[e] == "SM_4") {
    
    #If the camera trap date says 2017, add the offset
    if (year(DateTime2[e]) == "2017") {
      TrueDateTime <- DateTime2[e] + duration(Offset_SM4, units = "days")
    } else {
      TrueDateTime <- DateTime2[e]
    }
    #Round date to nearest minute and convert to character strings of date and time
    TrueDateTime <- round_date(TrueDateTime, unit = "minute")
    TrueDateTime <- strsplit(as.character(TrueDateTime), split = " ")[[1]]
    
    DF$TrueDate[e] <- TrueDateTime[1]
    DF$TrueTime[e] <- TrueDateTime[2]
  } else if (DF$Location[e] == "SM_5") {
    
    #If the camera trap date says 2017, add the offset
    if (year(DateTime2[e]) == "2017") {
      TrueDateTime <- DateTime2[e] + duration(Offset_SM5, units = "days")
    } else {
      TrueDateTime <- DateTime2[e]
    }
    #Round date to nearest minute and convert to character strings of date and time
    TrueDateTime <- round_date(TrueDateTime, unit = "minute")
    TrueDateTime <- strsplit(as.character(TrueDateTime), split = " ")[[1]]
    
    DF$TrueDate[e] <- TrueDateTime[1]
    DF$TrueTime[e] <- TrueDateTime[2]
  }
}

write.csv(NewDF, "F:/Camera Trap Photos/Observations_Nov21_New.csv")
