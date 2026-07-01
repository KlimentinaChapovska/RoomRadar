const STEPS = [
  {
    n: '01', title: 'Data Audit & EDA',
    body: 'Hotel Reservations dataset (36,275 rows × 19 columns). No nulls. Key findings: 545 zero-price rows (excluded from regression), 139 zero-adult rows, 37 calendar-invalid dates. Spearman correlations identify lead_time as the strongest cancellation signal (|r| = 0.42).',
  },
  {
    n: '02', title: 'Feature Engineering',
    body: 'Engineered inside sklearn Pipelines to prevent leakage: total_nights, total_guests, has_children, log_lead_time, log_price, price_is_zero, is_weekend_stay, has_special_request, prior_cancel_rate, arrival_season, arrival_dow.',
  },
  {
    n: '03', title: 'Classification Pipeline',
    body: 'Stratified 60/20/20 split (RS=42). Four models: Dummy, Logistic Regression, Decision Tree, XGBoost. XGBoost tuned with 5-fold CV. SelectFromModel feature selection (14/39 features kept). Final test AUC 0.953.',
  },
  {
    n: '04', title: 'Probability Calibration',
    body: 'Isotonic regression calibration (manual implementation — sklearn 1.9 removed cv="prefit"). Brier score improves from 0.084 to 0.079. Threshold selected at 0.34 using F1-optimal search on the validation set only.',
  },
  {
    n: '05', title: 'Regression Pipeline',
    body: 'Zero-price rows excluded. Four models: Dummy, Ridge, Decision Tree, XGBoost. Raw target outperforms log1p for tuned XGBoost (val RMSE 17.04 vs 18.05). Production model: XGBoost on raw avg_price_per_room. Test R² 0.755, RMSE 16.24.',
  },
  {
    n: '06', title: 'Production API',
    body: 'FastAPI serves both models. Single request returns: calibrated cancellation probability, risk band, predicted room price, price comparison, and rule-based business recommendations. CORS configurable via CORS_ORIGINS env var.',
  },
];

const LIMITATIONS = [
  'Models are trained on 2017–2018 data from a single (unidentified) hotel. They may not generalise to other properties or time periods.',
  'Room prices are in unspecified price units. The dataset documentation uses $ notation but the currency is unverified.',
  'Regression R² 0.755 — 24.5% of price variance is unexplained. Complementary and Aviation segments have very few rows (391 and 125 respectively).',
  'The cancellation model shows train→val F1 gap of 0.10 (0.931 → 0.831), indicating modest overfitting. Val→test gap is only 0.005, confirming stable generalisation.',
  'current_room_price maps to avg_price_per_room in the cancellation model. If a booking has no confirmed price yet, this feature will be zero, which the model interprets as a "complimentary" booking signal.',
  'No protected-attribute analysis was performed. The model may propagate patterns in historical data that reflect past management decisions.',
];

export default function About() {
  return (
    <>
      <div className="page-hero">
        <div className="page-hero__inner">
          <p className="page-hero__eyebrow">About / Methodology</p>
          <h1>How RoomRadar was built</h1>
          <p style={{ marginTop: '.75rem', maxWidth: '58ch', color: 'var(--clr-brown-mid)' }}>
            A transparent account of the data, modelling decisions, limitations,
            and technology choices behind the two production models.
          </p>
        </div>
      </div>

      <section className="section">
        <div className="container">
          {/* Dataset */}
          <div className="section-header text-center" style={{ marginBottom: '2.5rem' }}>
            <span className="section-header__eyebrow">Dataset</span>
            <h2>Hotel Reservations (2017–2018)</h2>
            <p>
              Publicly available hotel booking dataset. 36,275 reservations, 19 raw features,
              no missing values, no duplicates. Target: booking_status (Canceled / Not_Canceled).
            </p>
          </div>

          <div className="grid-4" style={{ marginBottom: '4rem' }}>
            {[
              { v: '36,275', l: 'Total reservations' },
              { v: '32.8%',  l: 'Cancellation rate' },
              { v: '105.1',  l: 'Avg price (price units)' },
              { v: '2017–18',l: 'Date range' },
            ].map(({ v, l }) => (
              <div key={l} className="kpi-card text-center">
                <div className="kpi-card__value">{v}</div>
                <div className="kpi-card__label" style={{ textAlign: 'center' }}>{l}</div>
              </div>
            ))}
          </div>

          {/* Pipeline steps */}
          <h2 style={{ marginBottom: '2rem' }}>Methodology</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '4rem' }}>
            {STEPS.map(({ n, title, body }) => (
              <article key={n} className="card" style={{ display: 'grid', gridTemplateColumns: '60px 1fr', gap: '1.25rem', alignItems: 'start' }}>
                <div style={{ fontFamily: 'var(--ff-serif)', fontSize: '1.75rem', fontWeight: 700, color: 'var(--clr-primary)', lineHeight: 1 }}>
                  {n}
                </div>
                <div>
                  <h3 style={{ marginBottom: '.5rem' }}>{title}</h3>
                  <p style={{ fontSize: '.9rem' }}>{body}</p>
                </div>
              </article>
            ))}
          </div>

          {/* Tech Stack */}
          <h2 style={{ marginBottom: '1.5rem' }}>Technology Stack</h2>
          <div className="grid-3" style={{ marginBottom: '4rem' }}>
            {[
              { layer: 'ML / Data', items: ['Python 3.13', 'scikit-learn 1.9', 'XGBoost 3.3', 'pandas, numpy', 'joblib (model serialisation)'] },
              { layer: 'Backend', items: ['FastAPI 0.115', 'Pydantic v2', 'Uvicorn', 'CORS middleware', 'Rotating file logs'] },
              { layer: 'Frontend', items: ['Vite 5 + React 18', 'React Router 6', 'Recharts 2', 'Custom SVG gauge', 'No UI library'] },
            ].map(({ layer, items }) => (
              <article key={layer} className="card">
                <h3 style={{ marginBottom: '1rem' }}>{layer}</h3>
                <ul style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                  {items.map(i => (
                    <li key={i} style={{ fontSize: '.875rem', color: 'var(--clr-text-muted)', display: 'flex', gap: '.5rem' }}>
                      <span style={{ color: 'var(--clr-primary)' }}>·</span> {i}
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>

          {/* Limitations */}
          <h2 style={{ marginBottom: '1.5rem' }}>Known Limitations</h2>
          <div className="alert alert-warning" style={{ marginBottom: '1rem' }}>
            <strong>These limitations are intentionally disclosed.</strong> RoomRadar is an academic project.
            Before using predictions in production, validate against your property's own data.
          </div>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
            {LIMITATIONS.map((l, i) => (
              <li key={i} className="card card-sm" style={{ display: 'flex', gap: '.75rem', fontSize: '.9rem', color: 'var(--clr-text-muted)' }}>
                <span style={{ color: 'var(--clr-warning)', fontWeight: 700, flexShrink: 0 }}>!</span>
                {l}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </>
  );
}
