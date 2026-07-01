# Hotel Reservations — Data Dictionary

| Column | Type | Description | Notes |
|---|---|---|---|
| Booking_ID | str | Unique booking identifier | Drop before training |
| no_of_adults | int | Number of adults | 0 in 139 rows — suspicious |
| no_of_children | int | Number of children | Max 10; 3 rows > 5 |
| no_of_weekend_nights | int | Weekend nights booked | |
| no_of_week_nights | int | Weekday nights booked | 78 rows with 0 total nights |
| type_of_meal_plan | str | Meal plan selected | Meal Plan 3 has only 5 rows |
| required_car_parking_space | int (0/1) | Parking requested | |
| room_type_reserved | str | Room type category | Room_Type 3 has only 7 rows |
| lead_time | int | Days between booking and arrival | 102 rows > 400 days |
| arrival_year | int | Year of arrival | Only 2017–2018 |
| arrival_month | int | Month of arrival (1–12) | |
| arrival_date | int | Day of month (1–31) | |
| market_segment_type | str | Booking channel | |
| repeated_guest | int (0/1) | Whether guest has stayed before | |
| no_of_previous_cancellations | int | Prior cancellations by guest | ⚠ Potential leakage risk |
| no_of_previous_bookings_not_canceled | int | Prior completed stays | ⚠ Potential leakage risk |
| avg_price_per_room | float | Average price per room per night | 545 zero-price rows; 1069 IQR outliers; max $540 |
| no_of_special_requests | int | Number of special requests | |
| booking_status | str | **Target** — Canceled / Not_Canceled | 67.2% Not_Canceled, 32.8% Canceled |

## Leakage Notes
- `no_of_previous_cancellations` and `no_of_previous_bookings_not_canceled` describe historical behaviour
  *of the guest*, not of the booking being predicted. They are known at booking time and are **not leakage**,
  but should be verified against business context.
- `Booking_ID` must be dropped — it encodes no signal and could leak row ordering.
- `avg_price_per_room` — for the **cancellation** model this feature could reflect post-booking adjustments;
  flag for discussion. For the **price** model it is the target, so exclude from price-model features.
