import { useState, useId } from 'react';
import { api } from '../services/api.js';
import { ErrorBoundary } from '../components/ErrorBoundary.jsx';
import { calcNights, parseArrival } from '../utils/dateHelpers.js';

/* ── Risk gauge (SVG arc) ──────────────────────────────────────────────── */
function RiskGauge({ probability }) {
  const pct   = Math.min(Math.max(probability ?? 0, 0), 1);
  const deg   = pct * 180;
  const r     = 80;
  const cx    = 110; const cy = 100;
  const startX = cx - r; const startY = cy;
  const endX   = cx + r; const endY   = cy;
  const rad    = (deg - 180) * (Math.PI / 180);
  const nx     = cx + r * Math.cos(rad);
  const ny     = cy + r * Math.sin(rad);
  const color  = pct < 0.30 ? '#4CAF50' : pct < 0.60 ? '#F59E0B' : '#EF4444';

  return (
    <div className="risk-gauge-wrap" role="img" aria-label={`Cancellation probability: ${(pct * 100).toFixed(1)}%`}>
      <svg viewBox="0 0 220 130" aria-hidden="true">
        {/* Background arc */}
        <path d={`M ${startX} ${cy} A ${r} ${r} 0 0 1 ${endX} ${cy}`} fill="none" stroke="#EEE6DE" strokeWidth="14" strokeLinecap="round" />
        {/* Zone arcs */}
        {[
          { from: 0,   to: 0.30, c: '#4CAF50' },
          { from: 0.30, to: 0.60, c: '#F59E0B' },
          { from: 0.60, to: 1.00, c: '#EF4444' },
        ].map(({ from, to, c }) => {
          const aR = (from * 180 - 180) * Math.PI / 180;
          const bR = (to   * 180 - 180) * Math.PI / 180;
          const ax = cx + r * Math.cos(aR); const ay = cy + r * Math.sin(aR);
          const bx = cx + r * Math.cos(bR); const by = cy + r * Math.sin(bR);
          const large = (to - from) > 0.5 ? 1 : 0;
          return (
            <path key={c} d={`M ${ax} ${ay} A ${r} ${r} 0 ${large} 1 ${bx} ${by}`}
              fill="none" stroke={c} strokeWidth="14" strokeLinecap="round" opacity=".25" />
          );
        })}
        {/* Active arc */}
        {pct > 0 && (
          <path d={`M ${startX} ${cy} A ${r} ${r} 0 0 1 ${nx} ${ny}`}
            fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
        )}
        {/* Needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="6" fill={color} />
        {/* Labels */}
        <text x={startX - 4} y={cy + 18} fontSize="10" fill="#AE9080" textAnchor="middle">0%</text>
        <text x={cx}         y={cy - r - 8} fontSize="10" fill="#AE9080" textAnchor="middle">50%</text>
        <text x={endX + 4}   y={cy + 18} fontSize="10" fill="#AE9080" textAnchor="middle">100%</text>
        {/* Probability text */}
        <text x={cx} y={cy + 28} fontSize="22" fontWeight="700" fill={color} textAnchor="middle"
          fontFamily="'Playfair Display', serif">
          {(pct * 100).toFixed(1)}%
        </text>
        <text x={cx} y={cy + 44} fontSize="10" fill="#AE9080" textAnchor="middle">cancel probability</text>
      </svg>
    </div>
  );
}

/* ── Defaults ───────────────────────────────────────────────────────────── */
const DEFAULT_CHECKIN  = '';
const DEFAULT_CHECKOUT = '';

const DEFAULTS = {
  no_of_adults:                         '',
  no_of_children:                       '',
  type_of_meal_plan:                    '',
  required_car_parking_space:           '',
  room_type_reserved:                   '',
  lead_time:                            '',
  market_segment_type:                  '',
  repeated_guest:                       '',
  no_of_previous_cancellations:         '',
  no_of_previous_bookings_not_canceled: '',
  current_room_price:                   '',
  no_of_special_requests:               '',
};

/* ── Field helpers ──────────────────────────────────────────────────────── */
const MEAL_PLANS = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'];
const ROOM_TYPES = ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'];
const SEGMENTS   = ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'];

function Field({ label, hint, children, id }) {
  return (
    <div className="form-group">
      <label className="form-label" htmlFor={id}>{label}</label>
      {children}
      {hint && <span className="form-hint">{hint}</span>}
    </div>
  );
}

function NumInput({ id, value, onChange, min, max, step = 1, placeholder }) {
  return (
    <input
      id={id} type="number" className="form-control"
      value={value} min={min} max={max} step={step}
      placeholder={placeholder}
      onChange={e => onChange(e.target.value)}
    />
  );
}

function Select({ id, value, onChange, options, placeholder }) {
  return (
    <select id={id} className="form-control" value={value} onChange={e => onChange(e.target.value)}>
      <option value="">{placeholder ?? '— select —'}</option>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

/* ── Band badge helper ──────────────────────────────────────────────────── */
function bandCls(band) {
  return band === 'high' ? 'badge-high' : band === 'medium' ? 'badge-medium' : 'badge-low';
}

/* ── Nights breakdown display ───────────────────────────────────────────── */
function NightsBreakdown({ nights }) {
  if (nights.total === 0) return null;
  return (
    <div className="nights-breakdown" aria-live="polite">
      <span className="nights-total">
        {nights.total} night{nights.total !== 1 ? 's' : ''}
      </span>
      <span className="nights-detail">
        {nights.weekday} weekday · {nights.weekend} weekend
      </span>
    </div>
  );
}

/* ── Main page ──────────────────────────────────────────────────────────── */
export default function Prediction() {
  const [form, setForm]           = useState({ ...DEFAULTS });
  const [checkIn, setCheckIn]     = useState(DEFAULT_CHECKIN);
  const [checkOut, setCheckOut]   = useState(DEFAULT_CHECKOUT);
  const [dateError, setDateError] = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [errorStatus, setErrorStatus] = useState(null);
  const uid = useId();

  const nights     = calcNights(checkIn, checkOut);
  const validDates = checkIn && checkOut && nights.total > 0;

  const set = (key) => (val) => setForm(f => ({ ...f, [key]: val }));

  const handleCheckIn = (val) => {
    setCheckIn(val);
    if (val && checkOut && new Date(val + 'T00:00:00') >= new Date(checkOut + 'T00:00:00')) {
      setDateError('Check-out must be after check-in.');
    } else {
      setDateError(null);
    }
  };

  const handleCheckOut = (val) => {
    setCheckOut(val);
    if (checkIn && val && new Date(checkIn + 'T00:00:00') >= new Date(val + 'T00:00:00')) {
      setDateError('Check-out must be after check-in.');
    } else {
      setDateError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validDates) {
      setDateError('Please select a valid check-in and check-out date.');
      return;
    }
    setLoading(true);
    setError(null);
    setErrorStatus(null);
    setResult(null);
    try {
      const n = (key) => Number(form[key]);
      const payload = {
        no_of_adults:                         n('no_of_adults'),
        no_of_children:                       n('no_of_children'),
        no_of_special_requests:               n('no_of_special_requests'),
        no_of_previous_cancellations:         n('no_of_previous_cancellations'),
        no_of_previous_bookings_not_canceled: n('no_of_previous_bookings_not_canceled'),
        current_room_price:                   n('current_room_price'),
        lead_time:                            n('lead_time'),
        required_car_parking_space:           n('required_car_parking_space'),
        repeated_guest:                       n('repeated_guest'),
        type_of_meal_plan:                    form.type_of_meal_plan,
        room_type_reserved:                   form.room_type_reserved,
        market_segment_type:                  form.market_segment_type,
        ...parseArrival(checkIn),
        no_of_weekend_nights:                 nights.weekend,
        no_of_week_nights:                    nights.weekday,
      };
      const data = await api.predict(payload);
      setResult(data);
    } catch (err) {
      setError(err.message ?? 'Prediction failed. Is the backend running?');
      setErrorStatus(err.status ?? null);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm({ ...DEFAULTS });
    setCheckIn(DEFAULT_CHECKIN);
    setCheckOut(DEFAULT_CHECKOUT);
    setDateError(null);
    setResult(null);
    setError(null);
    setErrorStatus(null);
  };

  return (
    <ErrorBoundary>
      <div className="page-hero">
        <div className="page-hero__inner">
          <p className="page-hero__eyebrow">Live Prediction</p>
          <h1>Predict cancellation risk &amp; room price</h1>
          <p style={{ marginTop: '.75rem', color: 'var(--clr-brown-mid)', maxWidth: '58ch' }}>
            Fill in the booking details below. The API returns a calibrated cancellation
            probability, risk band, predicted room price, and actionable recommendations.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '2rem', alignItems: 'start' }}>

            {/* ── Form ────────────────────────────────────────────── */}
            <form onSubmit={handleSubmit} noValidate aria-label="Booking prediction form">

              {/* Guests */}
              <div style={{ marginBottom: '2rem' }}>
                <div className="form-section-title">Guests</div>
                <div className="form-grid-3">
                  <Field label="Adults" id={`${uid}-adults`}>
                    <NumInput id={`${uid}-adults`} value={form.no_of_adults} onChange={set('no_of_adults')} min={1} max={4} placeholder="e.g. 2" />
                  </Field>
                  <Field label="Children" id={`${uid}-children`}>
                    <NumInput id={`${uid}-children`} value={form.no_of_children} onChange={set('no_of_children')} min={0} max={10} placeholder="e.g. 0" />
                  </Field>
                  <Field label="Special requests" id={`${uid}-special`}>
                    <NumInput id={`${uid}-special`} value={form.no_of_special_requests} onChange={set('no_of_special_requests')} min={0} max={5} placeholder="e.g. 0" />
                  </Field>
                </div>
              </div>

              {/* Stay */}
              <div style={{ marginBottom: '2rem' }}>
                <div className="form-section-title">Stay Dates &amp; Lead Time</div>
                <div className="form-grid-2">
                  <Field label="Check-in date" id={`${uid}-checkin`}>
                    <input
                      id={`${uid}-checkin`} type="date" className="form-control"
                      value={checkIn}
                      onChange={e => handleCheckIn(e.target.value)}
                      required
                    />
                  </Field>
                  <Field label="Check-out date" id={`${uid}-checkout`}>
                    <input
                      id={`${uid}-checkout`} type="date" className="form-control"
                      value={checkOut}
                      min={checkIn || undefined}
                      onChange={e => handleCheckOut(e.target.value)}
                      required
                    />
                  </Field>
                </div>

                {dateError && (
                  <div className="alert alert-error" style={{ marginTop: '.5rem', padding: '.5rem .75rem', fontSize: '.875rem' }} role="alert">
                    {dateError}
                  </div>
                )}

                <NightsBreakdown nights={nights} />

                <div className="form-grid-2" style={{ marginTop: '1.25rem' }}>
                  <Field label="Lead time (days)" id={`${uid}-lead`} hint="Days between booking and arrival">
                    <NumInput id={`${uid}-lead`} value={form.lead_time} onChange={set('lead_time')} min={0} placeholder="e.g. 45" />
                  </Field>
                  <Field label="Current room price" id={`${uid}-price`} hint="Price units — used for comparison">
                    <NumInput id={`${uid}-price`} value={form.current_room_price} onChange={set('current_room_price')} min={0} step={0.01} placeholder="e.g. 100" />
                  </Field>
                </div>
              </div>

              {/* Booking details */}
              <div style={{ marginBottom: '2rem' }}>
                <div className="form-section-title">Booking Details</div>
                <div className="form-grid-2">
                  <Field label="Meal plan" id={`${uid}-meal`}>
                    <Select id={`${uid}-meal`} value={form.type_of_meal_plan} onChange={set('type_of_meal_plan')} options={MEAL_PLANS} placeholder="— select meal plan —" />
                  </Field>
                  <Field label="Room type" id={`${uid}-room`}>
                    <Select id={`${uid}-room`} value={form.room_type_reserved} onChange={set('room_type_reserved')} options={ROOM_TYPES} placeholder="— select room type —" />
                  </Field>
                  <Field label="Market segment" id={`${uid}-seg`}>
                    <Select id={`${uid}-seg`} value={form.market_segment_type} onChange={set('market_segment_type')} options={SEGMENTS} placeholder="— select segment —" />
                  </Field>
                  <Field label="Car parking" id={`${uid}-park`}>
                    <Select id={`${uid}-park`} value={form.required_car_parking_space} onChange={set('required_car_parking_space')} options={[0, 1]} placeholder="No / Yes (0 / 1)" />
                  </Field>
                </div>
              </div>

              {/* History */}
              <div style={{ marginBottom: '2rem' }}>
                <div className="form-section-title">Guest History</div>
                <div className="form-grid-3">
                  <Field label="Repeated guest" id={`${uid}-rep`}>
                    <Select id={`${uid}-rep`} value={form.repeated_guest} onChange={set('repeated_guest')} options={[0, 1]} placeholder="No / Yes (0 / 1)" />
                  </Field>
                  <Field label="Prev. cancellations" id={`${uid}-pcanc`}>
                    <NumInput id={`${uid}-pcanc`} value={form.no_of_previous_cancellations} onChange={set('no_of_previous_cancellations')} min={0} placeholder="e.g. 0" />
                  </Field>
                  <Field label="Prev. completed stays" id={`${uid}-pstay`}>
                    <NumInput id={`${uid}-pstay`} value={form.no_of_previous_bookings_not_canceled} onChange={set('no_of_previous_bookings_not_canceled')} min={0} placeholder="e.g. 0" />
                  </Field>
                </div>
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button type="submit" className="btn btn-primary" disabled={loading} aria-busy={loading}>
                  {loading ? (
                    <><span className="spinner" style={{ width: 18, height: 18, border: '2px solid rgba(255,255,255,.4)', borderTopColor: 'white' }} aria-hidden="true" /> Predicting…</>
                  ) : 'Run Prediction →'}
                </button>
                <button type="button" className="btn btn-ghost" onClick={handleReset} disabled={loading}>
                  Reset
                </button>
              </div>

              {/* API error */}
              {error && (
                <div className="alert alert-error" style={{ marginTop: '1rem' }} role="alert">
                  <strong>{errorStatus === 422 ? 'Validation error:' : 'Error:'}</strong> {error}
                  {(!errorStatus || errorStatus >= 500) && (
                    <div style={{ marginTop: '.5rem', fontSize: '.8rem' }}>
                      Make sure the backend is running and <code>VITE_API_URL</code> is set.
                    </div>
                  )}
                </div>
              )}
            </form>

            {/* ── Result panel ─────────────────────────────────────── */}
            {result && (
              <aside aria-label="Prediction result" aria-live="polite">
                <div className="card" style={{ position: 'sticky', top: 'calc(var(--nav-h) + 1rem)' }}>
                  <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>Prediction Result</h2>

                  {/* Risk gauge + label */}
                  <RiskGauge probability={result.cancellation_probability} />
                  <div style={{ textAlign: 'center', marginTop: '.75rem', marginBottom: '1.25rem' }}>
                    <span className={`badge ${bandCls(result.cancellation_risk_band)}`} style={{ fontSize: '.9rem', padding: '.35rem 1rem' }}>
                      {result.cancellation_risk_band.toUpperCase()} RISK
                    </span>
                    <span style={{ marginLeft: '.75rem', fontFamily: 'var(--ff-serif)', fontWeight: 700, fontSize: '1.05rem', color: 'var(--clr-brown)' }}>
                      {result.cancellation_label.replace('_', ' ')}
                    </span>
                  </div>

                  {/* Price comparison */}
                  <div className="card card-sm" style={{ background: 'var(--clr-bg-alt)', marginBottom: '1.25rem' }}>
                    <div className="price-cmp-row">
                      <div>
                        <div className="price-cmp-label">Current price</div>
                        <div className="price-cmp-val">{result.price_comparison.current_room_price.toFixed(1)}</div>
                      </div>
                      <div className="price-cmp-vs">→</div>
                      <div>
                        <div className="price-cmp-label">Predicted price</div>
                        <div className="price-cmp-val" style={{ color: 'var(--clr-primary)' }}>
                          {result.price_comparison.predicted_room_price.toFixed(1)}
                        </div>
                      </div>
                    </div>
                    <p style={{ marginTop: '.75rem', fontSize: '.8rem' }}>
                      {result.price_comparison.note}
                    </p>
                  </div>

                  {/* Recommendations */}
                  {result.recommendations?.length > 0 && (
                    <div style={{ marginBottom: '1.25rem' }}>
                      <h3 style={{ fontSize: '1rem', marginBottom: '.75rem' }}>Recommendations</h3>
                      <ul className="recommendation-list">
                        {result.recommendations.map((rec, i) => (
                          <li key={i} className="recommendation-item">{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Meta */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.75rem', marginBottom: '.75rem' }}>
                    <div className="result-stat">
                      <div className="result-stat__label">Probability</div>
                      <div className="result-stat__value">{(result.cancellation_probability * 100).toFixed(1)}%</div>
                    </div>
                    <div className="result-stat">
                      <div className="result-stat__label">Response time</div>
                      <div className="result-stat__value">{result.prediction_time_ms?.toFixed(0)} ms</div>
                    </div>
                  </div>

                  <div style={{ fontSize: '.75rem', color: 'var(--clr-text-light)', marginTop: '.5rem' }}>
                    Request ID: <code style={{ fontSize: '.72rem' }}>{result.request_id}</code><br />
                    Models: clf v{result.model_versions?.cancellation} · price v{result.model_versions?.price}
                  </div>
                </div>
              </aside>
            )}
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}
