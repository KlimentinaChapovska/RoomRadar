/**
 * Pre-extracted analysis data for the Executive Dashboard.
 * Source: outputs/tables/ — values are hard-coded so the browser
 * never loads the raw CSV (which stays in data/raw/ on the server).
 */

export const kpis = {
  totalBookings: 36275,
  cancellationRate: 32.76,       // %
  avgRoomPrice: 105.12,           // price units
  repeatGuestRate: 2.0,           // % (1 - 33.58 new-guest cancel vs 1.72 repeat)
  totalCanceled: 11885,
  totalNotCanceled: 24390,
};

/** Cancel rate % by market segment */
export const cancellationBySegment = [
  { segment: 'Online',        rate: 36.5, count: 23214 },
  { segment: 'Offline',       rate: 29.9, count: 10528 },
  { segment: 'Aviation',      rate: 29.6, count: 125 },
  { segment: 'Corporate',     rate: 10.9, count: 2017 },
  { segment: 'Complementary', rate: 0.0,  count: 391 },
];

/** Cancel rate % by room type */
export const cancellationByRoomType = [
  { room: 'Room Type 1', rate: 32.3, count: 28130 },
  { room: 'Room Type 2', rate: 32.9, count: 692 },
  { room: 'Room Type 4', rate: 34.2, count: 6057 },
  { room: 'Room Type 5', rate: 27.2, count: 265 },
  { room: 'Room Type 6', rate: 42.0, count: 966 },
  { room: 'Room Type 7', rate: 22.8, count: 158 },
];

/** Cancel rate % by number of special requests */
export const cancellationBySpecialRequests = [
  { requests: '0', rate: 43.2, count: 19777 },
  { requests: '1', rate: 23.8, count: 11373 },
  { requests: '2', rate: 14.6, count: 4364 },
  { requests: '3+', rate: 0.0,  count: 761 },
];

/** Monthly booking volume and cancel rate */
export const monthlyTrend = [
  { label: 'Jul 17', volume: 363,  cancelRate: 66.9, avgPrice: null },
  { label: 'Aug 17', volume: 1014, cancelRate: 18.2, avgPrice: null },
  { label: 'Sep 17', volume: 1649, cancelRate: 11.0, avgPrice: null },
  { label: 'Oct 17', volume: 1913, cancelRate: 15.8, avgPrice: null },
  { label: 'Nov 17', volume: 647,  cancelRate: 4.2,  avgPrice: null },
  { label: 'Dec 17', volume: 928,  cancelRate: 2.4,  avgPrice: null },
  { label: 'Jan 18', volume: 1014, cancelRate: 2.4,  avgPrice: null },
  { label: 'Feb 18', volume: 1704, cancelRate: 25.2, avgPrice: null },
  { label: 'Mar 18', volume: 2358, cancelRate: 29.7, avgPrice: null },
  { label: 'Apr 18', volume: 2736, cancelRate: 36.4, avgPrice: null },
  { label: 'May 18', volume: 2598, cancelRate: 36.5, avgPrice: null },
  { label: 'Jun 18', volume: 3203, cancelRate: 40.3, avgPrice: null },
  { label: 'Jul 18', volume: 2557, cancelRate: 41.9, avgPrice: null },
  { label: 'Aug 18', volume: 2799, cancelRate: 46.6, avgPrice: null },
  { label: 'Sep 18', volume: 2962, cancelRate: 45.8, avgPrice: null },
  { label: 'Oct 18', volume: 3404, cancelRate: 46.4, avgPrice: null },
  { label: 'Nov 18', volume: 2333, cancelRate: 36.3, avgPrice: null },
  { label: 'Dec 18', volume: 2093, cancelRate: 18.2, avgPrice: null },
];

/** Repeat guest impact */
export const repeatGuestImpact = [
  { label: 'New Guest',    rate: 33.6 },
  { label: 'Repeat Guest', rate: 1.7 },
];

/** Lead time buckets and cancel rates (approximated from EDA) */
export const leadTimeBuckets = [
  { bucket: '0–14 d',   rate: 12.3 },
  { bucket: '15–30 d',  rate: 18.5 },
  { bucket: '31–60 d',  rate: 28.4 },
  { bucket: '61–90 d',  rate: 38.1 },
  { bucket: '91–150 d', rate: 47.2 },
  { bucket: '150+ d',   rate: 58.6 },
];

/** Booking distribution (pie) */
export const bookingDistribution = [
  { name: 'Not Canceled', value: 24390, pct: 67.24 },
  { name: 'Canceled',     value: 11885, pct: 32.76 },
];

/** Price comparison across room types (approximate from data) */
export const priceByRoomType = [
  { room: 'Type 1', avgPrice: 91 },
  { room: 'Type 2', avgPrice: 115 },
  { room: 'Type 4', avgPrice: 128 },
  { room: 'Type 5', avgPrice: 149 },
  { room: 'Type 6', avgPrice: 181 },
  { room: 'Type 7', avgPrice: 203 },
];
