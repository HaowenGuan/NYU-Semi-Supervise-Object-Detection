import matplotlib.pyplot as plt
import numpy as np

with open('./log.txt', 'r') as f:
    lines = f.readlines()

x = []
y = []
checkpoint = []
AP = []
AP50 = []
AP75 = []
APs = []
APm = []
APl = []
prev = None
for line in lines:
    line = line.split()

    if len(line) == 8 and line[5] == 'checkpoint':
        checkpoint.append(int(line[-1][-11:-4]))

    # if len(line) == 13 and line[0] == '|' and line[1] != 'AP' and line[2] == '|':
    if prev and prev[0] == '|:-----:|:------:|:------:|:-----:|:-----:|:-----:|':
        AP.append(float(line[1]))
        AP50.append(float(line[3]))
        AP75.append(float(line[5]))
        APs.append(float(line[7]))
        APm.append(float(line[9]))
        APl.append(float(line[11]))

    prev = line

    if len(line) > 9 and line[6] == 'iter:':
        x.append(int(line[7]))
        y.append(float(line[9]))

#%%
offset = 20
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(x[offset:], y[offset:])
ax1.set_yscale('log')
ax1.title.set_text("Training Loss")
ax1.set_xlabel('iter')
ax1.set_ylabel('loss')

ax2.plot(checkpoint, AP, label='AP')
ax2.plot(checkpoint, AP50, label='AP50')
ax2.plot(checkpoint, AP75, label='AP75')
ax2.plot(checkpoint, APs, label='APs')
ax2.plot(checkpoint, APm, label='APm')
ax2.plot(checkpoint, APl, label='APl')
ax2.legend()
ax2.title.set_text("Training AP")
ax2.set_xlabel('iter')
ax2.set_ylabel('AP')
#%%
