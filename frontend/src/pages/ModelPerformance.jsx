import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell,
} from 'recharts';
import {
  classificationLeaderboard,
  regressionLeaderboard,
  clfFeatureImportance,
  regFeatureImportance,
  riskBands,
  overfitSummary,
  calibration,
  confusionMatrix,
} from '../data/modelMetrics.js';

const CORAL  = '#C8694A';
const SAND   = '#D4B896';
const BROWN  = '#6B4530';

const fmt2 = v => (v * 100).toFixed(1) + '%';
const fmtN = v => typeof v === 'number' ? v.toFixed(3) : '—';

function MetricBadge({ value, good = true }) {
  return (
    <span className={good ? 'metric-good' : 'metric-bad'}>
      {typeof value === 'number' ? value.toFixed(3) : '—'}
    </span>
  );
}

export default function ModelPerformance() {
  const [cm] = [confusionMatrix.matrix];
  const [[tn, fp], [fn, tp]] = cm;

  return (
    <>
      <div className="page-hero">
        <div className="page-hero__inner">
          <p className="page-hero__eyebrow">Model Performance</p>
          <h1>How well do the models perform?</h1>
          <p style={{ marginTop: '.75rem', color: 'var(--clr-brown-mid)', maxWidth: '58ch' }}>
            All metrics are computed on a held-out test set the models never saw during training or tuning.
            Validation metrics guided hyperparameter search; test metrics measure true generalisation.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="container">

          {/* ── Classification leaderboard ───────────────────────── */}
          <h2 style={{ marginBottom: '.5rem' }}>Classification Leaderboard</h2>
          <p className="text-muted text-small" style={{ marginBottom: '1.5rem' }}>
            Predicts whether a booking will be cancelled. Positive class = Canceled. Decision threshold = 0.34.
          </p>
          <div className="table-wrap" style={{ marginBottom: '3rem' }}>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>AUC-ROC</th>
                  <th>Brier</th>
                </tr>
              </thead>
              <tbody>
                {classificationLeaderboard.map(m => (
                  <tr key={m.model} className={m.isProduction ? 'table-prod' : ''}>
                    <td>
                      {m.model}
                      {m.isProduction && <span className="badge badge-primary" style={{ marginLeft: '.5rem' }}>Production</span>}
                    </td>
                    <td><MetricBadge value={m.accuracy} /></td>
                    <td><MetricBadge value={m.precision} /></td>
                    <td><MetricBadge value={m.recall} /></td>
                    <td><MetricBadge value={m.f1} /></td>
                    <td><MetricBadge value={m.auc} /></td>
                    <td>{m.brier ? <MetricBadge value={m.brier} good={false} /> : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Regression leaderboard ───────────────────────── */}
          <h2 style={{ marginBottom: '.5rem' }}>Regression Leaderboard</h2>
          <p className="text-muted text-small" style={{ marginBottom: '1.5rem' }}>
            Predicts avg_price_per_room (price units). Baseline = predict mean. Raw target (no log1p).
          </p>
          <div className="table-wrap" style={{ marginBottom: '3rem' }}>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>R²</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                </tr>
              </thead>
              <tbody>
                {regressionLeaderboard.map(m => (
                  <tr key={m.model} className={m.isProduction ? 'table-prod' : ''}>
                    <td>
                      {m.model}
                      {m.isProduction && <span className="badge badge-primary" style={{ marginLeft: '.5rem' }}>Production</span>}
                    </td>
                    <td><MetricBadge value={m.r2} /></td>
                    <td style={{ color: 'var(--clr-text-muted)' }}>{m.mae.toFixed(2)}</td>
                    <td style={{ color: 'var(--clr-text-muted)' }}>{m.rmse.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Confusion matrix + calibration ───────────────────── */}
          <div className="chart-grid" style={{ marginBottom: '3rem' }}>
            <div className="chart-card">
              <div className="chart-card__title">Confusion Matrix (XGBoost, test set)</div>
              <p className="text-small text-muted" style={{ marginBottom: '1.25rem' }}>
                Predicted → columns; Actual → rows. Threshold = 0.34.
              </p>
              <div style={{ display: 'flex', gap: '2rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: '.5rem', alignItems: 'center', fontSize: '.85rem' }}>
                    <div />
                    <div style={{ textAlign: 'center', fontWeight: 600, color: 'var(--clr-brown-mid)', fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.04em' }}>Pred Not Cancel</div>
                    <div style={{ textAlign: 'center', fontWeight: 600, color: 'var(--clr-brown-mid)', fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.04em' }}>Pred Cancel</div>
                    <div style={{ fontWeight: 600, color: 'var(--clr-brown-mid)', fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.04em', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual Not Cancel</div>
                    <div className="conf-cell conf-tn">
                      <div className="conf-cell__n">{tn.toLocaleString()}</div>
                      <div className="conf-cell__label">True Negative</div>
                    </div>
                    <div className="conf-cell conf-fp">
                      <div className="conf-cell__n">{fp.toLocaleString()}</div>
                      <div className="conf-cell__label">False Positive</div>
                    </div>
                    <div style={{ fontWeight: 600, color: 'var(--clr-brown-mid)', fontSize: '.78rem', textTransform: 'uppercase', letterSpacing: '.04em', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual Cancel</div>
                    <div className="conf-cell conf-fn">
                      <div className="conf-cell__n">{fn.toLocaleString()}</div>
                      <div className="conf-cell__label">False Negative</div>
                    </div>
                    <div className="conf-cell conf-tp">
                      <div className="conf-cell__n">{tp.toLocaleString()}</div>
                      <div className="conf-cell__label">True Positive</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Probability Calibration (Brier Score)</div>
              <p className="text-small text-muted" style={{ marginBottom: '1rem' }}>Lower Brier = better calibration.</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={calibration} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="method" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0.07, 0.09]} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(3)} />
                  <Tooltip formatter={v => v.toFixed(4)} />
                  <Bar dataKey="brier" radius={[4, 4, 0, 0]} name="Brier Score">
                    {calibration.map((c, i) => (
                      <Cell key={i} fill={c.method === 'Isotonic' ? CORAL : SAND} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* ── Train/Val/Test overfitting ───────────────────────── */}
          <div className="chart-grid" style={{ marginBottom: '3rem' }}>
            <div className="chart-card">
              <div className="chart-card__title">Overfitting Check — F1 Across Splits</div>
              <p className="text-small text-muted" style={{ marginBottom: '1rem' }}>
                Train–Val gap = 0.10 (expected with XGBoost). Val–Test gap = 0.005 (stable generalisation).
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overfitSummary} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" />
                  <XAxis dataKey="split" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0.7, 1.0]} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
                  <Tooltip formatter={v => v.toFixed(3)} />
                  <Bar dataKey="f1" fill={CORAL} radius={[4, 4, 0, 0]} name="F1">
                    {overfitSummary.map((s, i) => (
                      <Cell key={i} fill={s.split === 'Test' ? BROWN : CORAL} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Risk Band Distribution (test set)</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem', marginTop: '.5rem' }}>
                {riskBands.map(({ band, pct, actualCancelRate, color }) => (
                  <div key={band}>
                    <div className="flex-between" style={{ marginBottom: '.3rem' }}>
                      <span style={{ fontSize: '.875rem', fontWeight: 600, color: 'var(--clr-brown)' }}>{band}</span>
                      <span style={{ fontSize: '.8rem', color: 'var(--clr-text-muted)' }}>{pct}% of bookings · {actualCancelRate}% actually cancel</span>
                    </div>
                    <div style={{ height: '10px', background: 'var(--clr-bg-alt)', borderRadius: '5px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: '5px', transition: 'width .5s' }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Feature importance ───────────────────────── */}
          <div className="chart-grid">
            <div className="chart-card">
              <div className="chart-card__title">Top 10 Features — Cancellation Model (gain)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={clfFeatureImportance} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={130} />
                  <Tooltip formatter={v => v.toFixed(3)} />
                  <Bar dataKey="importance" fill={CORAL} radius={[0, 4, 4, 0]} name="Gain importance" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <div className="chart-card__title">Top 10 Features — Price Model (gain)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={regFeatureImportance} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEE6DE" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={130} />
                  <Tooltip formatter={v => v.toFixed(3)} />
                  <Bar dataKey="importance" fill={SAND} stroke={BROWN} strokeWidth={1} radius={[0, 4, 4, 0]} name="Gain importance" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
