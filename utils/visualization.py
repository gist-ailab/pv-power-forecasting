import matplotlib.pyplot as plt


plt.plot(batch_x.cpu()[0])
cols = ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
       'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
plt.legend(cols)
plt.savefig('batch_x.png')


plt.clf()
plt.plot(batch_x_mark.cpu()[0])
stamp_cols = ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']
plt.legend(stamp_cols)
plt.savefig('batch_x_mark.png')

plt.plot(batch_y.cpu()[0])
plt.legend(cols)
plt.savefig('batch_y.png')


plt.clf()
plt.plot(batch_y_mark.cpu()[0])
plt.legend(stamp_cols)
plt.savefig('batch_y_mark.png')


