import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="flex-center" style={{ minHeight: 'calc(100vh - var(--nav-h))', flexDirection: 'column', gap: '1.5rem', padding: '3rem 1.5rem', textAlign: 'center' }}>
      <div style={{ fontFamily: 'var(--ff-serif)', fontSize: '6rem', fontWeight: 700, color: 'var(--clr-apricot)', lineHeight: 1 }}>
        404
      </div>
      <h1 style={{ fontSize: '1.75rem' }}>Page not found</h1>
      <p style={{ maxWidth: '40ch' }}>
        The page you were looking for doesn't exist or has moved.
      </p>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
        <Link to="/" className="btn btn-primary">Back to Home</Link>
        <Link to="/predict" className="btn btn-outline">Try Prediction</Link>
      </div>
    </div>
  );
}
