import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="footer" aria-label="Site footer">
      <div className="footer__inner">
        <div className="footer__brand">
          <h3 className="footer__heading">RoomRadar</h3>
          <p>Spot risk. Price smarter. Manage better.</p>
          <p style={{ marginTop: '.5rem', fontSize: '.8rem' }}>
            ML-powered hotel intelligence — cancellation risk + room price prediction.
          </p>
        </div>

        <nav aria-label="Footer navigation">
          <h4 className="footer__heading">Pages</h4>
          <ul className="footer__links">
            <li><Link to="/">Home</Link></li>
            <li><Link to="/dashboard">Executive Dashboard</Link></li>
            <li><Link to="/predict">Live Prediction</Link></li>
            <li><Link to="/performance">Model Performance</Link></li>
            <li><Link to="/about">About / Methodology</Link></li>
          </ul>
        </nav>

        <div>
          <h4 className="footer__heading">Stack</h4>
          <ul className="footer__links">
            <li><span>FastAPI + XGBoost backend</span></li>
            <li><span>Calibrated classification</span></li>
            <li><span>XGBoost regression</span></li>
            <li><span>Vite + React + Recharts</span></li>
          </ul>
        </div>
      </div>

      <div className="footer__bottom">
        <span>© {new Date().getFullYear()} RoomRadar — Academic project</span>
        <span>Models trained on Hotel Reservations dataset · Prices in unspecified units</span>
      </div>
    </footer>
  );
}
