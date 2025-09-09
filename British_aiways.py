import numpy as np
import pandas as pd

df = pd.read_excel('C:/Users/vikto/Downloads/British_Airways_Summer_Schedule_Dataset_Forage_Data_Science_Task (1).xlsx')
df = df.drop(['FLIGHT_DATE', 'FLIGHT_TIME', 'AIRLINE_CD', 'FLIGHT_NO', 'DEPARTURE_STATION_CD', 'ARRIVAL_STATION_CD', 'ARRIVAL_COUNTRY', 'AIRCRAFT_TYPE'], axis=1)
group_df = df.groupby(by=["TIME_OF_DAY", "HAUL", "ARRIVAL_REGION"]).sum().reset_index()
group_df ['TIER_1'] = round(((group_df['TIER1_ELIGIBLE_PAX'])*100)/(group_df[['FIRST_CLASS_SEATS', 'BUSINESS_CLASS_SEATS', 'ECONOMY_SEATS']].sum(axis=1)),2)
group_df ['TIER_2'] = round(((group_df['TIER2_ELIGIBLE_PAX'])*100)/(group_df[['FIRST_CLASS_SEATS', 'BUSINESS_CLASS_SEATS', 'ECONOMY_SEATS']].sum(axis=1)),2)
group_df ['TIER_3'] = round(((group_df['TIER3_ELIGIBLE_PAX'])*100)/(group_df[['FIRST_CLASS_SEATS', 'BUSINESS_CLASS_SEATS', 'ECONOMY_SEATS']].sum(axis=1)),2)
new_df = group_df.drop(['TIER1_ELIGIBLE_PAX', 'TIER2_ELIGIBLE_PAX', 'TIER3_ELIGIBLE_PAX', 'FIRST_CLASS_SEATS', 'BUSINESS_CLASS_SEATS', 'ECONOMY_SEATS'], axis=1)

Q1_FIRST_CLASS_SEATS = group_df['FIRST_CLASS_SEATS'].quantile(0.25)
Q3_FIRST_CLASS_SEATS = group_df['FIRST_CLASS_SEATS'].quantile(0.75)
mean_FIRST_CLASS_SEATS = np.mean(group_df['FIRST_CLASS_SEATS'])
median_FIRST_CLASS_SEATS = np.median(group_df['FIRST_CLASS_SEATS'])

Q1_BUSINESS_CLASS_SEATS = group_df['BUSINESS_CLASS_SEATS'].quantile(0.25)
Q3_BUSINESS_CLASS_SEATS = group_df['BUSINESS_CLASS_SEATS'].quantile(0.75)
mean_BUSINESS_CLASS_SEATS = np.mean(group_df['BUSINESS_CLASS_SEATS'])
median_BUSINESS_CLASS_SEATS = np.median(group_df['BUSINESS_CLASS_SEATS'])

Q1_ECONOMY_SEATS = group_df['ECONOMY_SEATS'].quantile(0.25)
Q3_ECONOMY_SEATS = group_df['ECONOMY_SEATS'].quantile(0.75)
mean_ECONOMY_SEATS = np.mean(group_df['ECONOMY_SEATS'])
median_ECONOMY_SEATS = np.median(group_df['ECONOMY_SEATS'])

def assign_notes(row):
    if row['HAUL'] == "SHORT":
        if Q3_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS']: 
            return "Короткие рейсы, нет пассажиров First Class. В TIER_1 пассажиры преимущественно BA Gold Guest List и держатели BA Premier Card. Пассажиров Business class много. "
        elif Q3_BUSINESS_CLASS_SEATS > row['BUSINESS_CLASS_SEATS'] > mean_BUSINESS_CLASS_SEATS:
            return "Короткие рейсы, нет пассажиров First Class. В TIER_1 пассажиры преимущественно BA Gold Guest List и держатели BA Premier Card. Пассажиров Business class выше среднего. "
        elif Q1_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS'] < mean_BUSINESS_CLASS_SEATS : 
            return "Короткие рейсы, нет пассажиров First Class. В TIER_1 пассажиры преимущественно BA Gold Guest List и держатели BA Premier Card. Пассажиров Business class ниже среднего."
        elif row['BUSINESS_CLASS_SEATS'] < Q1_BUSINESS_CLASS_SEATS: 
            return "Короткие рейсы, нет пассажиров First Class. В TIER_1 пассажиры преимущественно BA Gold Guest List и держатели BA Premier Card. Пассажиров Business class мало."
        else:
            return None
    elif row['HAUL'] == "LONG":
        if Q1_FIRST_CLASS_SEATS < row['FIRST_CLASS_SEATS'] < mean_FIRST_CLASS_SEATS:
            if Q3_BUSINESS_CLASS_SEATS > row['BUSINESS_CLASS_SEATS'] > mean_BUSINESS_CLASS_SEATS:
                return "Длинные рейсы, пассажиров First Class меньше среднего. Пассажиров Business class выше среднего."
            elif Q3_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS']:
                return "Длинные рейсы, пассажиров First Class меньше среднего. Пассажиров Business class много."
            elif Q1_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS'] < mean_BUSINESS_CLASS_SEATS :
                return "Длинные рейсы, пассажиров First Class меньше среднего. Пассажиров Business class ниже среднего."
            elif row['BUSINESS_CLASS_SEATS'] < Q1_BUSINESS_CLASS_SEATS: 
                return "Длинные рейсы, пассажиров First Class меньше среднего. Пассажиров Business class мало."
            else:
                return None
        elif row['FIRST_CLASS_SEATS'] < Q1_FIRST_CLASS_SEATS :
            if Q3_BUSINESS_CLASS_SEATS > row['BUSINESS_CLASS_SEATS'] > mean_BUSINESS_CLASS_SEATS:
                return "Длинные рейсы, пассажиров First Class мало. Пассажиров Business class выше среднего."
            elif Q3_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS']:
                return "Длинные рейсы, пассажиров First Class мало. Пассажиров Business class много."
            elif Q1_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS'] < mean_BUSINESS_CLASS_SEATS :
                return "Длинные рейсы, пассажиров First Class мало. Пассажиров Business class ниже среднего."
            elif row['BUSINESS_CLASS_SEATS'] < Q1_BUSINESS_CLASS_SEATS: 
                return "Длинные рейсы, пассажиров First Class мало. Пассажиров Business class мало."
            else:
                return None
        elif Q3_FIRST_CLASS_SEATS > row['FIRST_CLASS_SEATS'] > mean_FIRST_CLASS_SEATS:
            if Q3_BUSINESS_CLASS_SEATS > row['BUSINESS_CLASS_SEATS'] > mean_BUSINESS_CLASS_SEATS:
                return "Длинные рейсы, пассажиров First Class выше среднего. Пассажиров Business class выше среднего."
            elif Q3_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS']:
                return "Длинные рейсы, пассажиров First Class выше среднего. Пассажиров Business class много."
            elif Q1_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS'] < mean_BUSINESS_CLASS_SEATS :
                return "Длинные рейсы, пассажиров First Class выше среднего. Пассажиров Business class ниже среднего."
            elif row['BUSINESS_CLASS_SEATS'] < Q1_BUSINESS_CLASS_SEATS: 
                return "Длинные рейсы, пассажиров First Class выше среднего. Пассажиров Business class мало."
            else:
                return None
        elif Q3_FIRST_CLASS_SEATS < row['FIRST_CLASS_SEATS']:
            if Q3_BUSINESS_CLASS_SEATS > row['BUSINESS_CLASS_SEATS'] > mean_BUSINESS_CLASS_SEATS:
                return "Длинные рейсы, пассажиров First Class много. Пассажиров Business class выше среднего."
            elif Q3_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS']:
                return "Длинные рейсы, пассажиров First Class много. Пассажиров Business class много."
            elif Q1_BUSINESS_CLASS_SEATS < row['BUSINESS_CLASS_SEATS'] < mean_BUSINESS_CLASS_SEATS :
                return "Длинные рейсы, пассажиров First Class много. Пассажиров Business class ниже среднего."
            elif row['BUSINESS_CLASS_SEATS'] < Q1_BUSINESS_CLASS_SEATS: 
                return "Длинные рейсы, пассажиров First Class много. Пассажиров Business class мало."
            else:
                return None        
        else:
            return None 
    else:
        return None

group_df['Notes'] = group_df.apply(assign_notes, axis=1)

def ECONOMY_SEATS_notes(row):
    if Q3_ECONOMY_SEATS >= row['ECONOMY_SEATS'] >= mean_ECONOMY_SEATS:
        return "Пассажиров Econom class выше среднего."
    elif Q1_ECONOMY_SEATS < row['ECONOMY_SEATS'] < mean_ECONOMY_SEATS:
        return "Пассажиров Econom class меньше среднего."
    elif Q1_ECONOMY_SEATS > row['ECONOMY_SEATS']:
        return "Пассажиров Econom class мало."
    elif Q3_ECONOMY_SEATS <= row['ECONOMY_SEATS']:
        return "Пассажиров Econom class много."
    
group_df['Notes_ECONOMY_SEATS'] = group_df.apply(ECONOMY_SEATS_notes, axis=1)
new_df['N'] = new_df['TIME_OF_DAY'] + ', ' + new_df['HAUL']
new_df['Notes'] = group_df['Notes'] + ' ' + group_df['Notes_ECONOMY_SEATS']
new_df = new_df.sort_values(by='TIER_1', ascending=False)

new_df.to_excel('DONE.xlsx', index=False)


