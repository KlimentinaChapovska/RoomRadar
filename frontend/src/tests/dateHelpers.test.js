import { describe, it, expect } from 'vitest';
import { calcNights, parseArrival } from '../utils/dateHelpers.js';

// Reference: Jan 1 2024 = Monday
describe('calcNights — validation', () => {
  it('returns zero totals for same check-in and check-out', () => {
    expect(calcNights('2024-01-08', '2024-01-08')).toEqual({ weekday: 0, weekend: 0, total: 0 });
  });

  it('returns zero when check-out is before check-in', () => {
    expect(calcNights('2024-01-10', '2024-01-09')).toEqual({ weekday: 0, weekend: 0, total: 0 });
  });

  it('returns zero for empty strings', () => {
    expect(calcNights('', '')).toEqual({ weekday: 0, weekend: 0, total: 0 });
  });

  it('returns zero when only check-in is provided', () => {
    expect(calcNights('2024-01-08', '')).toEqual({ weekday: 0, weekend: 0, total: 0 });
  });
});

describe('calcNights — single night', () => {
  it('counts Monday night as one weekday', () => {
    // 2024-01-08 = Monday
    expect(calcNights('2024-01-08', '2024-01-09')).toEqual({ weekday: 1, weekend: 0, total: 1 });
  });

  it('counts Friday night as one weekday', () => {
    // 2024-01-12 = Friday
    expect(calcNights('2024-01-12', '2024-01-13')).toEqual({ weekday: 1, weekend: 0, total: 1 });
  });

  it('counts Saturday night as one weekend', () => {
    // 2024-01-13 = Saturday
    expect(calcNights('2024-01-13', '2024-01-14')).toEqual({ weekday: 0, weekend: 1, total: 1 });
  });

  it('counts Sunday night as one weekend', () => {
    // 2024-01-14 = Sunday
    expect(calcNights('2024-01-14', '2024-01-15')).toEqual({ weekday: 0, weekend: 1, total: 1 });
  });
});

describe('calcNights — multi-night stays', () => {
  it('splits a Thu–Mon stay: 2 weekday + 2 weekend', () => {
    // Nights: Thu(wd), Fri(wd), Sat(we), Sun(we); checkout Mon Jan 15
    expect(calcNights('2024-01-11', '2024-01-15')).toEqual({ weekday: 2, weekend: 2, total: 4 });
  });

  it('splits a full-week Mon–Mon stay: 5 weekday + 2 weekend', () => {
    // Nights: Mon–Fri (5 weekday), Sat–Sun (2 weekend); checkout Mon Jan 15
    expect(calcNights('2024-01-08', '2024-01-15')).toEqual({ weekday: 5, weekend: 2, total: 7 });
  });

  it('counts a weekend-only Sat–Mon stay: 0 weekday + 2 weekend', () => {
    // Nights: Sat Jan 13, Sun Jan 14; checkout Mon Jan 15
    expect(calcNights('2024-01-13', '2024-01-15')).toEqual({ weekday: 0, weekend: 2, total: 2 });
  });

  it('counts a midweek Tue–Fri stay: 3 weekday + 0 weekend', () => {
    // Nights: Tue Jan 9, Wed Jan 10, Thu Jan 11; checkout Fri Jan 12
    expect(calcNights('2024-01-09', '2024-01-12')).toEqual({ weekday: 3, weekend: 0, total: 3 });
  });

  it('handles a 14-night stay spanning two full weeks', () => {
    // 10 weekday + 4 weekend
    expect(calcNights('2024-01-08', '2024-01-22')).toEqual({ weekday: 10, weekend: 4, total: 14 });
  });
});

describe('parseArrival', () => {
  it('extracts year, month, and day correctly', () => {
    expect(parseArrival('2018-06-15')).toEqual({ arrival_year: 2018, arrival_month: 6, arrival_date: 15 });
  });

  it('handles January (month 1) without zero-padding issues', () => {
    expect(parseArrival('2024-01-01')).toEqual({ arrival_year: 2024, arrival_month: 1, arrival_date: 1 });
  });

  it('handles December 31', () => {
    expect(parseArrival('2023-12-31')).toEqual({ arrival_year: 2023, arrival_month: 12, arrival_date: 31 });
  });

  it('returns numeric values, not strings', () => {
    const result = parseArrival('2024-03-07');
    expect(typeof result.arrival_year).toBe('number');
    expect(typeof result.arrival_month).toBe('number');
    expect(typeof result.arrival_date).toBe('number');
  });
});
