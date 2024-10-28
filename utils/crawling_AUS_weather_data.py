import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Initialize Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Set date range for data collection
location_code = 15590
start_date = datetime(2008, 1, 1)
end_date = datetime(2024, 10, 27)
current_date = start_date
total_days = (end_date - start_date).days + 1

# List to store failed dates
failed_dates = []

# File name for storing data
file_name = 'Alice_Springs_weather_data.csv'

# Initialize CSV file if starting fresh
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'Weather_Relative_Humidity', 'Wind_Speed'])

# List to accumulate monthly data
monthly_data = []

# Loop through each date to collect data
for _ in tqdm(range(total_days), desc="Progress", unit="day"):
    date_str = current_date.strftime('%Y-%m-%d')
    url = f'https://www.weatherzone.com.au/station/SITE/{location_code}/observations/{date_str}'
    driver.get(url)

    try:
        # Wait until the table is loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'tbody'))
        )

        # Get page source once the JavaScript content is loaded
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Look for the table without relying on dynamically changing classes
        table_body = soup.find('tbody')
        if table_body:
            rows = table_body.find_all('tr')[1:]  # Skip the first row with the next day's data
            daily_data = []

            for row in rows:
                time_tag = row.find('td', class_='hourly-obs-date')
                humidity_tag = row.find('td', class_='hourly-obs-humidityt')
                wind_speed_tag = row.find('td', class_='hourly-obs-windSpeed')

                if time_tag and humidity_tag and wind_speed_tag:
                    time_str = time_tag.get_text(strip=True)
                    time_str = " ".join(time_str.split()[1:])  # Get time only, skipping weekday
                    time_str = time_str.replace(" ACST", "").replace(" ACDT", "")  # Remove timezone
                    try:
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                        humidity_str = humidity_tag.get_text(strip=True)
                        wind_speed_str = wind_speed_tag.get_text(strip=True)
                    except ValueError as ve:
                        print(f"Failed to parse time for {date_str}: {ve}")
                        failed_dates.append(f"{date_str} {time_str}")

            # Sort daily data in ascending order by timestamp
            daily_data.sort(key=lambda x: x[0])

            # Accumulate daily data into monthly_data list
            monthly_data.extend(daily_data)
        else:
            print(f"No data table found for {date_str}")
            failed_dates.append(date_str)

    except Exception as e:
        print(f"Failed to retrieve data for {date_str}: {e}")
        failed_dates.append(date_str)

    # Move to the next day
    current_date += timedelta(days=1)

    # Save accumulated data at the end of each month
    if current_date.day == 1 or current_date > end_date:
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(monthly_data)
        monthly_data = []

# Close the driver
driver.quit()

# Print failed dates
if failed_dates:
    print("\nFailed to retrieve data for the following timestamps:")
    for date in failed_dates:
        print(date)

print("Data collection complete.")
