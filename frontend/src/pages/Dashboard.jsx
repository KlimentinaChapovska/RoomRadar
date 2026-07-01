import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend,
} from 'recharts';
import {
  kpis,
  cancellationBySegment,
  cancellationByRoomType,
  cancellationBySpecialRequests,
  monthlyTrend,
  repeatGuestImpact,
  leadTimeBuckets,
  bookingDistribution,
  priceByRoomType,
} from '../data/dashboardData.js';

const CORAL  = '#C8694A';
const ROSE   = '#C08080';
const SAND   = '#D4B896';
const BROWN  = '#6B4530';
const APRICOT= '#E8C4A0';

const KPIs = [
  { label: 'Total Bookings',     value: kpis.totalBookings.toLocaleString(), sub: '2017–2018' },
  { label: 'Cancellation Rate',  value: `${kpis.cancellationRate}%`, sub: `${kpis.totalCanceled.toLocaleString()} canceled`, accent: true },
  { label: 'Avg Room Price',     value: `${kpis.avgRoomPrice}`, sub: 'price units', },
  { label: 'Repeat Guest Rate',  value: `${kpis.repeatGuestRate}%`, sub: `${Math.round(kpis.totalBookings * kpis.repeatGuestRate / 100)} repeat guests` },
];

const fmtPct = v => `${v}%`;

export default function Dashboard() {
  return (
    <>
      <div className="page-hero">
        <div className="page-hero__inner">
          <p className="page-hero__eyebrow">Executive Dashboard</p>
          <h1>Booking analytics overview</h1>
          <p style={{ marginTop: '.75rem', color: 'var(--clr-brown-mid)', maxWidth: '58ch' }}>
            Pre-extracted insights from the Hotel Reservations dataset (36,275 bookings, 2017–2018).
            No raw data is loaded in the browser.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="container">

          {/* KPI row */}
          <div className="grid-4" style={{ marginBottom: '3rem' }}>
            {KPIs.map(({ label, value, sub, accent }) => (
              <div key={label} className={`kpi-card${accent ? ' kpi-card--accent' : ''}`}>
                <div className="kpi-card__label">{label}</div>
                <div className="kpi-card__value">{value}</div>
                <div className="kpi-card__sub">{sub}</div>
              </div>
            ))}
          </div>

          {/* Row 1 — Booking distribution + Monthly trend */}
          <div className="chart-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="chart-card">
              <div className="chart-card__title">Booking Outcome Distribution</div>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie
                    data={bookingDistribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    label={({ name, pct }) => `${name} ${pct}%`}
                  >
                    <Cell fill={SAND} />
                    <Cell fill={CORAL} />
                  </Pie>
                  <Tooltip formatter={(v) => v.toLocaleString()} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Monthly Cancellation Rate (%)</div>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={monthlyTrend} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={fmtPct} domain={[0, 60]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={fmtPct} />
                  <Line type="monotone" dataKey="cancelRate" stroke={CORAL} strokeWidth={2} dot={false} name="Cancel rate" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Row 2 — Segment + Room Type */}
          <div className="chart-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="chart-card">
              <div className="chart-card__title">Cancellation Rate by Market Segment (%)</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={cancellationBySegment} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" horizontal={false} />
                  <XAxis type="number" tickFormatter={fmtPct} domain={[0, 50]} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="segment" tick={{ fontSize: 12 }} width={100} />
                  <Tooltip formatter={fmtPct} />
                  <Bar dataKey="rate" fill={CORAL} radius={[0, 4, 4, 0]} name="Cancel rate" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Cancellation Rate by Room Type (%)</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={cancellationByRoomType} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="room" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={fmtPct} domain={[0, 60]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={fmtPct} />
                  <Bar dataKey="rate" fill={BROWN} radius={[4, 4, 0, 0]} name="Cancel rate" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Row 3 — Special requests + Lead time */}
          <div className="chart-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="chart-card">
              <div className="chart-card__title">Cancellation Rate by Special Requests</div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={cancellationBySpecialRequests} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="requests" tick={{ fontSize: 12 }} />
                  <YAxis tickFormatter={fmtPct} domain={[0, 50]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={fmtPct} />
                  <Bar dataKey="rate" fill={ROSE} radius={[4, 4, 0, 0]} name="Cancel rate" />
                </BarChart>
              </ResponsiveContainer>
              <p style={{ fontSize: '.78rem', color: 'var(--clr-text-light)', marginTop: '.5rem' }}>
                Guests with zero special requests cancel at 43.2% — the model's third-strongest signal.
              </p>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Cancellation Rate by Lead Time Bucket (%)</div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={leadTimeBuckets} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={fmtPct} domain={[0, 60]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={fmtPct} />
                  <Bar dataKey="rate" fill={APRICOT} stroke={BROWN} strokeWidth={1} radius={[4, 4, 0, 0]} name="Cancel rate" />
                </BarChart>
              </ResponsiveContainer>
              <p style={{ fontSize: '.78rem', color: 'var(--clr-text-light)', marginTop: '.5rem' }}>
                Lead time is the model's strongest signal: Spearman |r| = 0.42 with cancellation.
              </p>
            </div>
          </div>

          {/* Row 4 — Repeat guest impact + Price by room type */}
          <div className="chart-grid">
            <div className="chart-card">
              <div className="chart-card__title">Cancellation Rate: New vs Repeat Guest</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={repeatGuestImpact} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                  <YAxis tickFormatter={fmtPct} domain={[0, 40]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={fmtPct} />
                  <Bar dataKey="rate" fill={CORAL} radius={[4, 4, 0, 0]} name="Cancel rate" />
                </BarChart>
              </ResponsiveContainer>
              <p style={{ fontSize: '.78rem', color: 'var(--clr-text-light)', marginTop: '.5rem' }}>
                Repeat guests cancel at 1.7% vs 33.6% for new guests — the clearest separation in the data.
              </p>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Avg Room Price by Room Type (price units)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={priceByRoomType} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="room" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="avgPrice" fill={SAND} stroke={BROWN} strokeWidth={1} radius={[4, 4, 0, 0]} name="Avg price" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
