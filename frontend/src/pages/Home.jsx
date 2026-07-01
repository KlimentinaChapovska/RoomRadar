import { Link } from 'react-router-dom';

const FEATURES = [
  {
    title: 'Cancellation Risk',
    desc: 'Calibrated XGBoost model with isotonic probability calibration. AUC 0.953, F1 0.826 on held-out test data.',
  },
  {
    title: 'Price Intelligence',
    desc: 'XGBoost regression predicts optimal room price. R² 0.755, RMSE 16.2 price units on the test set.',
  },
  {
    title: 'Risk Bands',
    desc: 'Three actionable tiers — Low, Medium, High — with business recommendations tailored to each booking.',
  },
  {
    title: 'Real-Time API',
    desc: 'FastAPI backend serves both models in a single request. Request IDs and structured logs included.',
  },
  {
    title: 'Honest Limitations',
    desc: 'Trained on 2017–2018 data. Predictions extrapolate beyond this window. Model limitations are documented.',
  },
  {
    title: 'Executive Dashboard',
    desc: 'EDA insights rendered as interactive Recharts — segment trends, seasonality, lead-time risk, and more.',
  },
];

export default function Home() {
  return (
    <>
      {/* Hero */}
      <section className="home-hero" aria-label="Hero">
        <div className="home-hero__inner">
          <span className="home-hero__eyebrow">
            <span aria-hidden="true">◎</span> ML-powered hotel intelligence
          </span>
          <h1 className="home-hero__title">Smarter hotel booking decisions</h1>
          <p className="home-hero__tagline">
            Spot risk. Price smarter. Manage better.
          </p>
          <p style={{ maxWidth: '52ch', marginBottom: '2rem', color: 'var(--clr-brown-mid)' }}>
            RoomRadar combines XGBoost cancellation prediction with room-price regression
            to give revenue managers a single, actionable intelligence layer on every booking.
          </p>
          <div className="home-hero__actions">
            <Link to="/predict" className="btn btn-primary">
              Try Live Prediction →
            </Link>
            <Link to="/dashboard" className="btn btn-outline">
              View Dashboard
            </Link>
          </div>

          <div className="home-stats" aria-label="Key statistics">
            <div className="home-stats__item">
              <div className="home-stats__num">0.953</div>
              <div className="home-stats__desc">ROC-AUC (cancellation)</div>
            </div>
            <div className="home-stats__item">
              <div className="home-stats__num">0.755</div>
              <div className="home-stats__desc">R² (price prediction)</div>
            </div>
            <div className="home-stats__item">
              <div className="home-stats__num">36 k</div>
              <div className="home-stats__desc">Hotel reservations analysed</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="section" aria-labelledby="features-heading">
        <div className="container">
          <div className="section-header">
            <span className="section-header__eyebrow">What RoomRadar does</span>
            <h2 id="features-heading">Prediction, intelligence, and transparency</h2>
            <p>
              Two production-grade ML models serve every request. The results are paired
              with business recommendations and presented in plain language.
            </p>
          </div>
          <div className="grid-3">
            {FEATURES.map(({ title, desc }) => (
              <article key={title} className="card">
                <h3 style={{ marginBottom: '.5rem' }}>{title}</h3>
                <p style={{ fontSize: '.9rem' }}>{desc}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* Risk band callout */}
      <section className="section-sm" style={{ background: 'var(--clr-bg-alt)', borderTop: '1px solid var(--clr-border)', borderBottom: '1px solid var(--clr-border)' }} aria-labelledby="bands-heading">
        <div className="container">
          <div className="section-header" style={{ marginBottom: '2rem' }}>
            <span className="section-header__eyebrow">Risk Bands</span>
            <h2 id="bands-heading">Three tiers. Three responses.</h2>
          </div>
          <div className="grid-3">
            {[
              { band: 'Low', range: '< 30 %', pct: 63.1, actual: 6.6, desc: 'No immediate action required.', cls: 'badge-low' },
              { band: 'Medium', range: '30 – 60 %', pct: 7.5, actual: 39.9, desc: 'Send pre-arrival confirmation.', cls: 'badge-medium' },
              { band: 'High', range: '≥ 60 %', pct: 29.4, actual: 87.3, desc: 'Request deposit or offer incentive.', cls: 'badge-high' },
            ].map(({ band, range, pct, actual, desc, cls }) => (
              <article key={band} className="card" style={{ textAlign: 'center' }}>
                <span className={`badge ${cls}`} style={{ marginBottom: '1rem' }}>{band}</span>
                <div style={{ fontFamily: 'var(--ff-serif)', fontSize: '1.1rem', fontWeight: 600, color: 'var(--clr-brown)', marginBottom: '.5rem' }}>
                  {range}
                </div>
                <div style={{ fontSize: '.8rem', color: 'var(--clr-text-muted)', marginBottom: '.75rem' }}>
                  {pct}% of bookings · {actual}% actually cancel
                </div>
                <p style={{ fontSize: '.875rem' }}>{desc}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section" aria-labelledby="cta-heading">
        <div className="container text-center">
          <h2 id="cta-heading">Ready to predict your next booking?</h2>
          <p style={{ marginTop: '.75rem', marginBottom: '2rem', maxWidth: '48ch', marginInline: 'auto' }}>
            Enter reservation details and receive a calibrated cancellation probability,
            predicted room price, and actionable recommendations — instantly.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/predict"     className="btn btn-primary">Open Prediction Form</Link>
            <Link to="/performance" className="btn btn-outline">See Model Performance</Link>
          </div>
        </div>
      </section>
    </>
  );
}
