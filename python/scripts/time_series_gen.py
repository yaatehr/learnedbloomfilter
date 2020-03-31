import datetime
import pandas as pd
from datetime import datetime
import random
import numpy as np
date_rng = pd.date_range(start='1/1/2010', end='4/01/2020', freq='s').to_frame(index=True)
# print(date_rng.head(10))
# date_samples = date_rng.sample(n=500) #sample 10 mil second timestamps (our of abour 30 mil)
date_samples = date_rng.sample(n=50000000) #sample 10 mil second timestamps (our of abour 30 mil)
# date_samples = date_rng.loc[random.sample(list(date_rng.index),500)]
# print(date_samples[0])

date_samples["busy"] = 0
# dinner_time = lambda row: row.between_time("18:15", "18:30") or row.between_time("18:30", "18:45") and random.getrandbits(1)
# jazz_theory = lambda row: row.between_time("14:00", "15:15") or (row.between_time("15:15", "15:30") and not random.getrandbits(2))
# monday_schedule = lambda row: dinner_time(row) or row.between_time('11:00', '12:30') or (row.between_time("14:00", "15:00") and random.getrandbits(1)) or (row.between_time("15:00", "16:30") and not random.getrandbits(5))
# wednesday_schedule = lambda row: dinner_time(row) or row.between_time('11:00', '12:30') or (row.between_time("15:00", "16:30") and not random.getrandbits(5))
# friday_schedule = lambda row: dinner_time(row) or row.between_time('11:00', '12:00')
# tuesday_schedule = lambda row: dinner_time(row) or (row.between_time("12:00", "13:00") and random.getrandbits(1)) or row.between_time("13:00", "13:30") or jazz_theory(row) or row.between_time("17:15", "19:15") or (row.between_time("16:00", "17:00") and not random.getrandbits(3))
# thursday_schedule = lambda row: dinner_time(row)  or jazz_theory(row) or row.between_time("17:15", "19:15") or row.between_time("16:00", "17:00") 
# saturday_schedule = lambda row: row.between_time("16:00", "19:00") or row.between_time("22:00", "23:00") and not random.getrandbits(2)
# sunday_schedule = lambda row: row.between_time("11:00", "11:30") or row.between_time("5:15", "6:00") or dinner_time(row) or row.between_time("19:00", "20:30")




# dinner_time = lambda row: row.between_time("18:15", "18:30").append(random.choice([row.between_time("18:30", "18:45"), None])) # if random.getrandbits(1) else None

dinner_time = lambda row: row.between_time("18:15", "18:30").append(row.between_time("18:30", "18:45")) # if random.getrandbits(1) else None
jazz_theory = lambda row: row.between_time("14:00", "15:15").append(row.between_time("15:15", "15:30"))
monday_schedule = lambda row: dinner_time(row).append(row.between_time('11:00', '12:30')).append(row.between_time("14:00", "15:00")).append(row.between_time("15:00", "16:30"))
wednesday_schedule = lambda row: dinner_time(row).append(row.between_time('11:00', '12:30')).append(row.between_time("15:00", "16:30"))
friday_schedule = lambda row: dinner_time(row).append(row.between_time('11:00', '12:00'))
tuesday_schedule = lambda row: dinner_time(row).append(row.between_time("12:00", "13:00")).append(row.between_time("13:00", "13:30")).append(jazz_theory(row)).append(row.between_time("17:15", "19:15")).append(row.between_time("16:00", "17:00"))
thursday_schedule = lambda row: dinner_time(row).append(jazz_theory(row)).append(row.between_time("17:15", "19:15")).append(row.between_time("16:00", "17:00"))
saturday_schedule = lambda row: row.between_time("16:00", "19:00").append(row.between_time("22:00", "23:00"))
sunday_schedule = lambda row: row.between_time("11:00", "11:30").append(row.between_time("5:15", "6:00")).append(dinner_time(row)).append(row.between_time("19:00", "20:30"))

# some time stamps are probabalistic
# dinner_time = lambda row: row.between_time("18:15", "18:30").append(row.between_time("18:30", "18:45") if random.getrandbits(1) else None)
# jazz_theory = lambda row: row.between_time("14:00", "15:15").append(row.between_time("15:15", "15:30"))
# monday_schedule = lambda row: dinner_time(row).append(row.between_time('11:00', '12:30')).append(row.between_time("14:00", "15:00")).append(row.between_time("15:00", "16:30"))
# wednesday_schedule = lambda row: dinner_time(row).append(row.between_time('11:00', '12:30')).append(row.between_time("15:00", "16:30") if random.getrandbits(5) else None)
# friday_schedule = lambda row: dinner_time(row).append(row.between_time('11:00', '12:00'))
# tuesday_schedule = lambda row: dinner_time(row).append(row.between_time("12:00", "13:00")).append(row.between_time("13:00", "13:30")).append(jazz_theory(row)).append(row.between_time("17:15", "19:15")).append(row.between_time("16:00", "17:00") if not random.getrandbits(3) else None)
# thursday_schedule = lambda row: dinner_time(row).append(jazz_theory(row)).append(row.between_time("17:15", "19:15")).append(row.between_time("16:00", "17:00"))
# saturday_schedule = lambda row: row.between_time("16:00", "19:00").append(row.between_time("22:00", "23:00"))
# sunday_schedule = lambda row: row.between_time("11:00", "11:30").append(row.between_time("5:15", "6:00")).append(dinner_time(row)).append(row.between_time("19:00", "20:30"))


schedules = [monday_schedule, tuesday_schedule, wednesday_schedule, thursday_schedule, friday_schedule, saturday_schedule, sunday_schedule]

# string_date_rng = [x for x in date_samples["timestamp"]]
# timestamp_date_rng = pd.to_datetime(string_date_rng, infer_datetime_format=True)


# date_samples["datetime"] = timestamp_date_rng

# monday_schedule_times = dinner_time(date_rng)
# print(monday_schedule_times.tail(10))

inset = pd.DataFrame()

for i, schedule in enumerate(schedules):
    if i == 0:
        inset = schedule(date_samples)
    else:
        inset.append(schedule(date_samples))

inset["busy"] = 1

# print(inset.head(10))
# print(date_samples.head(5))
ds1 = set([tuple(line) for line in inset.values])
ds2 = set([tuple(line) for line in date_samples.values])
# print(ds1)
# print(ds2)
outset = pd.DataFrame(list(ds2.difference(ds1)), columns=inset.columns)
# print(outset.head(2))

date_samples = inset.append(outset.head(len(inset)))
# for i, row in date_samples.iterrows():
#     dt = row["datetime"]
#     row["busy"] = day_of_week_schedules[dt.dayofweek](dt)
# print(date_samples.head(2))



date_samples.sample(frac=1).to_csv(r'time_series_attempt_v2.csv', index = False)
