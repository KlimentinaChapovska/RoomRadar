/**
 * Compute weekday and weekend nights between two ISO date strings.
 * A "night" is indexed by the date the guest sleeps (from check-in day up to,
 * but not including, check-out day). Saturday and Sunday nights are weekend.
 */
export function calcNights(checkIn, checkOut) {
  const ci = checkIn  ? new Date(checkIn  + 'T00:00:00') : null;
  const co = checkOut ? new Date(checkOut + 'T00:00:00') : null;

  if (!ci || !co || isNaN(ci) || isNaN(co) || co <= ci) {
    return { weekday: 0, weekend: 0, total: 0 };
  }

  let weekday = 0;
  let weekend = 0;
  const cur = new Date(ci);
  while (cur < co) {
    const dow = cur.getDay(); // 0=Sun, 6=Sat
    if (dow === 0 || dow === 6) weekend++;
    else weekday++;
    cur.setDate(cur.getDate() + 1);
  }
  return { weekday, weekend, total: weekday + weekend };
}

/**
 * Extract arrival_year, arrival_month, arrival_date from "YYYY-MM-DD".
 */
export function parseArrival(dateStr) {
  const [year, month, day] = dateStr.split('-').map(Number);
  return { arrival_year: year, arrival_month: month, arrival_date: day };
}
