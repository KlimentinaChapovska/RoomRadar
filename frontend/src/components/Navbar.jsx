import { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';

const LINKS = [
  { to: '/',           label: 'Home',        end: true },
  { to: '/dashboard',  label: 'Dashboard',   end: false },
  { to: '/predict',    label: 'Live Predict',end: false },
  { to: '/performance',label: 'Performance', end: false },
  { to: '/about',      label: 'About',       end: false },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="navbar" aria-label="Main navigation">
      <div className="navbar__inner">
        <Link to="/" className="navbar__brand" aria-label="RoomRadar home">
          <div className="navbar__logo-ring" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="8" stroke="white" strokeWidth="2"/>
              <circle cx="10" cy="10" r="4.5" stroke="white" strokeWidth="1.5"/>
              <circle cx="10" cy="10" r="1.5" fill="white"/>
              <line x1="10" y1="2" x2="10" y2="0" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <span className="navbar__name">RoomRadar</span>
        </Link>

        <button
          className="navbar__toggle"
          aria-expanded={open}
          aria-controls="nav-menu"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen(v => !v)}
        >
          {open ? '✕' : '☰'}
        </button>

        <ul
          id="nav-menu"
          className={`navbar__nav${open ? ' open' : ''}`}
          role="list"
        >
          {LINKS.map(({ to, label, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) => `navbar__link${isActive ? ' active' : ''}`}
                onClick={() => setOpen(false)}
              >
                {label}
              </NavLink>
            </li>
          ))}
          <li>
            <Link to="/predict" className="navbar__cta" onClick={() => setOpen(false)}>
              Try Prediction →
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}
