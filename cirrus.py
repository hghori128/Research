#%%

import numpy as np
import matplotlib.pyplot as plt
import math

import numpy.matlib as matlib
import datetime
import matplotlib

import pandas as pd
import netCDF4 as nc

import xarray as xr



#---------------------CORRECT COADD-----------
# Iniatalize coadd function to be used below 
# To add in height and time to inc
def coadd(q,z,layer):

    l = math.floor(len(q) / layer) * layer
    print(l)
    q = q[1:l+1]
    z = z[1:l+1]
    # Reshape q and z so that the bins from each layer are in the
    # same column
    qc = np.reshape(q, (layer,int(l/layer)), order='F')
    zc = np.reshape(z, (layer,int(l/layer)), order='F')
    print(np.shape(q))
    qc = (np.sum(qc,0))
    zc = (np.median(zc,0))
    return [qc, zc]

# %%
#FUNCTION TO OPEN FILEs, LOAD DATA ------------------------------------------------------------------
import glob

def open_file(year, month, day):
  """This function requires input of the filename for the particular day of interest
  this requires the year (XXXX) the month (XX) and the day (XX) IN STRING FORMAT
  i.e. April 4, 2020:
  year = '2020'
  month - '04'
  day - '04'
  """

  str1 = '/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/'

  str2 = year 
  str3 = month
  str4 = day
  str5 = '*.nc'
  filename = str1 +str2 + '/'+ str3 + '/'+ str4 +'/'+str5
  print(filename)
  
  #print(os.listdir(filename))
  
  if glob.glob(filename) :
    
    files = nc.MFDataset(filename)
    backscatter = files.variables['beta_raw'][:]
    clouds = files.variables['cbh'][:]
    cloud_variation = files.variables['cbe'][:]
    time_raw = files.variables['time'][:]
    range_raw= files.variables['range'][:]
    files.close()

  else:
    backscatter = np.zeros((5740, 1024))
    clouds = np.zeros((5740, 3))
    cloud_variation = np.zeros((5740, 3))
    time_raw = np.zeros((5740, 3))
    range_raw= np.zeros(1024)

   

  return [backscatter, clouds, time_raw, range_raw] 


#days to process (this is because python doesn't like 04 format):

days = [ '01', '02', '03', '04',  '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', 
  '15', '16', '17', '18', '19', '20' , '21','22','23', '24', '25', '26', '27',
  '28', '29', '30']
#days = ['21']

nstep = 30  #set equal to number of days to process

clouds = np.empty((nstep, 5740,3)) 
raw_backscatter = np.empty((nstep, 5740))

n = 0  #day counter 

#loop over days to process
for i in range(len(days)):

  [b, cloud, time_raw, height_range] = open_file('2020', '04', days[i])
  datasize = np.size(cloud[:,0])
  print(datasize)

  if datasize > 5739:

    clouds[n:,] = np.copy(cloud[0:5740:,0:3])
    raw_backscatter[n:,] = np.copy(b[0:5740:,0])
    n  += 1

  elif cloud.all == 0:
    clouds[n:,] = np.copy(0)
    raw_backscatter[n:,] = np.copy(0)
    
  else:
    diff = 5740 - datasize 
    newdata1 = np.pad(cloud, (0,diff), 'constant')
    newdata2 = np.pad(b, (0,diff), 'constant')
    clouds[n:,] = np.copy(newdata1[0:5740:,0:3])
    raw_backscatter[n:,] = np.copy(newdata2[0:5740:,0])
    n += 1



#%%




customdate = datetime.datetime(year=1904, month=1, day=1, hour=0,second=0)
realtime = [ customdate + datetime.timedelta(seconds=i) for i in (time_raw)]

utc = np.array([f.strftime('%H:%M') for f in (realtime)])

#pick out clouds that are reported as greater than 5000m in the cloud base 
# height variable 

istep = 0
cloud_5000m = np.empty((nstep, 5740,3)) #array to store cloud base heights >5000m

for i in range(len(days)):
  #append height values for clouds >5000m and 0 otherwise
  cloud5000 = np.where(clouds[istep]> 5000, clouds[istep], 0) 

  cloud_5000m[istep:,]= np.copy(cloud5000) #append to array for the day
  istep += 1

#%%
#PICKS OUT THE TIME SPANS OVER WHICH THERE IS CLOUD COVER ABOVE 5000m

istep = 0
cloud_time_ranges_l1 = []  #array of times where there are clouds >5000m in UTC TIME
cloud_time_ranges_l2 = []  #array of times where there are clouds >5000m in UTC TIME
cloud_time_ranges_l3 = []  #array of times where there are clouds >5000m in UTC TIME

cloud_backscatter = []  #array of raw backscatter only where there are clouds >5000m 
cloud_heights_l1 = []  #array of cloud heights only when >5000m in LAYER 1
cloud_heights_l2 = []    #array of cloud heights only when >5000m in LAYER 1
cloud_heights_l3 = []  #array of cloud heights only when >5000m in LAYER 1

for i in range(len(days)):
  print(istep)
  f1 = np.where(cloud_5000m[istep][:,0]>0) #clouds in layer 1 
  f2 = np.where(cloud_5000m[istep][:,1]>0) #clouds in layer 2 
  f3 = np.where(cloud_5000m[istep][:,2]>0) #clouds in layer 3
  print(np.shape(f1))

  print(np.shape(cloud_heights_l1))
 
  
  size = np.size(f1)
  print(size)

  if size > 0:  #if there ARE clouds >5000m do this:
    #cloud_heights_l1 = np.concatenate(cloud_5000m[istep][:,0][f1])
    cloud_heights_l1.append(cloud_5000m[istep][:,0][f1])
    cloud_heights_l2.append(cloud_5000m[istep][:,1][f2])
    cloud_heights_l3.append(cloud_5000m[istep][:,2][f3])

    print(np.shape(f1))
    timespan1 = utc[f1]            #currently takes instances in layer1
    timespan2 = utc[f2]            #currently takes instances in layer1
    timespan3 = utc[f3]            #currently takes instances in layer1
    
    #bs = raw_backscatter[istep][f1[0]]  #currently takes instances in layer 1
    cloud_time_ranges_l1.append(timespan1)  
    cloud_time_ranges_l2.append(timespan2)  
    cloud_time_ranges_l3.append(timespan3)  

    #cloud_backscatter.append(bs)
    

    istep += 1
  
  else:    #if no clouds >5000m, append the entire row as 0 
    cloud_time_ranges_l1.append(0)
    cloud_time_ranges_l2.append(0)
    cloud_time_ranges_l3.append(0)
    cloud_heights_l1.append(0)
    cloud_heights_l2.append(0)
    cloud_heights_l3.append(0)
    cloud_backscatter.append(0)
    istep += 1
    continue


#%%

#USE THIS TO PLOT TIME SPAN OF THE CLOUD 

import matplotlib.ticker as ticker
import xlsxwriter


for i in range(len(days)):

  size = np.size(cloud_time_ranges_l1[i])
  
  
  if size > 1:
    print(i)
    fig = plt.figure( )
    ax = fig.add_subplot(2,1,1)
    ax1 = fig.add_subplot(2,1,2, sharex=ax)
    #ax.plot( cloud_time_ranges[i], cloud_backscatter[i]/1000000)
    s = i+1
    #ax.xaxis.set_major_locator(ticker.LinearLocator(12))
    ax.set_ylabel(' Backscatter  (a.u)')
    ax.set_xlabel('UTC time (hh:mm)', fontsize='large')
    ax.set_title("Cirrus Cloud Backscatter and Heights on April %s 2017" %s)  

    print(np.shape(cloud_time_ranges_l1[i]))
    print(np.shape(cloud_heights_l1[i]))
    ax1.plot( cloud_time_ranges_l1[i], np.transpose(cloud_heights_l1[i]))
    ax1.scatter(cloud_time_ranges_l1[i], np.transpose(cloud_heights_l1[i]), label='cloud layer 1')
    
    #ax1.plot( cloud_time_ranges_l2[i], np.transpose(cloud_heights_l2[i]), lw=0.5)
    ax1.scatter(cloud_time_ranges_l2[i], np.transpose(cloud_heights_l2[i]), label='cloud layer 2')
    
    ax1.plot( cloud_time_ranges_l3[i], np.transpose(cloud_heights_l3[i]), lw=0.5, c='green')
    ax1.scatter(cloud_time_ranges_l3[i], np.transpose(cloud_heights_l3[i]), label='cloud layer 3')
    ax1.legend()
    ax1.xaxis.set_major_locator(ticker.LinearLocator(12))
    ax1.set_ylabel(' Cloud height (m)')
    ax1.set_xlabel('UTC time (hh:mm)', fontsize='large')
    #ax1.set_title("Cloud Backscatter on April %i 2020" %i)  

    #fig.savefig("cloud_plot{}".format(s))
    #plt.close()

    
    #workbook.close()
  
  else:
    print(i)

    continue

#%%
#CREATE EXCELL FILE HERE WITH ALL PLOTS----------------------------

workbook = xlsxwriter.Workbook('2017_april.xlsx')

worksheet = workbook.add_worksheet()

fstep = 0
for j in range(len(days)):

  size2 = np.size(cloud_time_ranges[j])
  print(np.shape(size2))

  #print(size2)
  if size2>=1:

    

    image = '/Users/hannanghori/Documents/university shit/4th year/Research/cloud_plot{}.png'.format(j)
   
  
    worksheet.insert_image(0,fstep,image, {'x_scale': 0.7, 'y_scale': 0.7})
    fstep += 1
    print(fstep)
  else:
    continue

workbook.close()




# %%

#FUNCTION TO MAKE APPROPRIATE PROFILE when given normal backscatter raw variable---------------------------------------------------------

def make_profile(data, time1, time2):
  """this function makes the appropriate backscatter profile given the input backscatter raw
  dataset for the particular day 
  data = input dataset
  time 1 = utc time in hours ('03:00')
  time 2 = utc time in hours ('03:00')"""

  overlap = np.loadtxt('data.txt',dtype = float)
  print(overlap)


  timespan1 = np.where(utc==time1)[0][0]
  timespan2 = np.where(utc==time2)[0][0]
  d_night_test1 = data[(timespan1):(timespan2),:] 

  d_nightsum_test1 = np.mean(d_night_test1,0)

  noise1 = np.mean(d_nightsum_test1[950:1024])
  print(noise1)
  d_nightsum_test1 = d_nightsum_test1 + noise1

  for f in overlap:
    dens_overlap_test1 = d_nightsum_test1 * overlap
  #If overlap is needed:
  [x1,y1] = coadd(dens_overlap_test1,height_range,5)
  return [x1,y1]

[x1,y1] = make_profile(b, '00:00', '06:00')


#%%
plt.plot(x1/10000000,y1)
plt.xlabel('Range Corrected Backscatter power (a.u)', fontsize='large')
plt.ylabel('Height (m)', fontsize='large')
plt.title('Backscatter Power over April 17 2020 (04:00 - 08:00)', fontsize='large')
plt.xlim([0,0.45])
plt.ylim([2000,15000])

# %%

grad = np.gradient(x1)
plt.plot(grad/10000000,y1)
plt.xlim([-0.07,0.07])
plt.ylim([2000,15000])
plt.xlabel('Gradient of Backscatter (a.u)', fontsize='large')
plt.ylabel('Height (m)', fontsize='large')
plt.title('Gradient of Backscatter Power over April 17 2020 (04:00 - 08:00)', fontsize='large')
# %%
from numpy import diff

dydx = diff(y1)/diff(x1)
plt.plot(dydx[0:203],y1[0:203])
#plt.xlim([-0.02,0.05])
plt.ylim([0,15000])
plt.xlabel('Range Corrected Backscatter power (a.u)', fontsize='large')
plt.ylabel('Height (m)', fontsize='large')
plt.title('Backscatter Power over April 17 2020 (04:00 - 08:00)', fontsize='large')

# %%
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix

ds = xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/04/25/*.nc',concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
df = ds.to_dataframe()
#d1 = np.array(files18_L2_1.variables['attenuated_backscatter_0'][:])

#%%

# %%
