import { describe, it, expect } from 'vitest';
import { validateBookingForm } from '../pages/Prediction.jsx';

const VALID_FORM = {
  no_of_adults:                         '2',
  no_of_children:                       '0',
  no_of_special_requests:               '0',
  type_of_meal_plan:                    'Meal Plan 1',
  required_car_parking_space:           '0',
  room_type_reserved:                   'Room_Type 1',
  lead_time:                            '30',
  market_segment_type:                  'Online',
  repeated_guest:                       '0',
  no_of_previous_cancellations:         '0',
  no_of_previous_bookings_not_canceled: '0',
  current_room_price:                   '100',
};
const VALID_NIGHTS = { weekday: 2, weekend: 0, total: 2 };

describe('validateBookingForm — clean form passes', () => {
  it('returns no errors for a fully filled form', () => {
    const errs = validateBookingForm(VALID_FORM, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(Object.keys(errs)).toHaveLength(0);
  });
});

describe('validateBookingForm — date validation', () => {
  it('flags missing check-in', () => {
    const errs = validateBookingForm(VALID_FORM, '', '2024-06-05', VALID_NIGHTS);
    expect(errs.checkIn).toBe('This field is required.');
  });

  it('flags missing check-out', () => {
    const errs = validateBookingForm(VALID_FORM, '2024-06-03', '', VALID_NIGHTS);
    expect(errs.checkOut).toBe('This field is required.');
  });

  it('flags check-out not after check-in', () => {
    const errs = validateBookingForm(VALID_FORM, '2024-06-05', '2024-06-03', { weekday: 0, weekend: 0, total: 0 });
    expect(errs.checkOut).toBe('Check-out must be after check-in.');
  });

  it('does not flag checkOut when check-in is missing too', () => {
    const errs = validateBookingForm(VALID_FORM, '', '', { weekday: 0, weekend: 0, total: 0 });
    expect(errs.checkIn).toBe('This field is required.');
    expect(errs.checkOut).toBe('This field is required.');
  });
});

describe('validateBookingForm — adults', () => {
  it('flags empty adults', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_adults: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_adults).toBe('This field is required.');
  });

  it('flags adults = 0', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_adults: '0' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_adults).toBe('Value must be at least 1.');
  });

  it('accepts adults = 1', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_adults: '1' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_adults).toBeUndefined();
  });
});

describe('validateBookingForm — required numeric fields', () => {
  it('flags empty children', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_children: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_children).toBe('This field is required.');
  });

  it('flags empty special requests', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_special_requests: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_special_requests).toBe('This field is required.');
  });

  it('flags empty lead time', () => {
    const errs = validateBookingForm({ ...VALID_FORM, lead_time: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.lead_time).toBe('This field is required.');
  });

  it('flags empty current room price', () => {
    const errs = validateBookingForm({ ...VALID_FORM, current_room_price: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.current_room_price).toBe('This field is required.');
  });

  it('flags empty previous cancellations', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_previous_cancellations: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_previous_cancellations).toBe('This field is required.');
  });

  it('flags empty previous bookings not canceled', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_previous_bookings_not_canceled: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_previous_bookings_not_canceled).toBe('This field is required.');
  });

  it('accepts zero for children', () => {
    const errs = validateBookingForm({ ...VALID_FORM, no_of_children: '0' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.no_of_children).toBeUndefined();
  });
});

describe('validateBookingForm — required select fields', () => {
  it('flags unselected meal plan', () => {
    const errs = validateBookingForm({ ...VALID_FORM, type_of_meal_plan: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.type_of_meal_plan).toBe('Please select a meal plan.');
  });

  it('flags unselected room type', () => {
    const errs = validateBookingForm({ ...VALID_FORM, room_type_reserved: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.room_type_reserved).toBe('Please select a room type.');
  });

  it('flags unselected market segment', () => {
    const errs = validateBookingForm({ ...VALID_FORM, market_segment_type: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.market_segment_type).toBe('Please select a market segment.');
  });

  it('flags unselected car parking', () => {
    const errs = validateBookingForm({ ...VALID_FORM, required_car_parking_space: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.required_car_parking_space).toBe('This field is required.');
  });

  it('flags unselected repeated guest', () => {
    const errs = validateBookingForm({ ...VALID_FORM, repeated_guest: '' }, '2024-06-03', '2024-06-05', VALID_NIGHTS);
    expect(errs.repeated_guest).toBe('This field is required.');
  });
});

describe('validateBookingForm — multiple errors at once', () => {
  it('reports all empty fields on a blank form', () => {
    const blank = Object.fromEntries(Object.keys(VALID_FORM).map(k => [k, '']));
    const errs = validateBookingForm(blank, '', '', { weekday: 0, weekend: 0, total: 0 });
    expect(errs.checkIn).toBeTruthy();
    expect(errs.checkOut).toBeTruthy();
    expect(errs.no_of_adults).toBeTruthy();
    expect(errs.no_of_children).toBeTruthy();
    expect(errs.type_of_meal_plan).toBeTruthy();
    expect(errs.room_type_reserved).toBeTruthy();
    expect(errs.market_segment_type).toBeTruthy();
    expect(Object.keys(errs).length).toBeGreaterThanOrEqual(10);
  });
});
