import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { ErrorBoundary } from '../components/ErrorBoundary.jsx';
import Home from '../pages/Home.jsx';
import NotFound from '../pages/NotFound.jsx';
import Navbar from '../components/Navbar.jsx';

// Recharts resize observer
global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };

describe('Home page', () => {
  it('renders hero heading', () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getByRole('heading', { name: /smarter hotel booking/i })).toBeTruthy();
  });

  it('renders primary CTA link', () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    const links = screen.getAllByRole('link', { name: /try live prediction/i });
    expect(links.length).toBeGreaterThan(0);
  });

  it('renders stat: AUC value', () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getByText('0.953')).toBeTruthy();
  });
});

describe('NotFound page', () => {
  it('renders 404 heading', () => {
    render(<MemoryRouter><NotFound /></MemoryRouter>);
    expect(screen.getByText('404')).toBeTruthy();
  });

  it('renders back-home link', () => {
    render(<MemoryRouter><NotFound /></MemoryRouter>);
    expect(screen.getByRole('link', { name: /back to home/i })).toBeTruthy();
  });
});

describe('Navbar', () => {
  it('renders brand name', () => {
    render(<MemoryRouter><Navbar /></MemoryRouter>);
    expect(screen.getByText('RoomRadar')).toBeTruthy();
  });

  it('has accessible nav landmark', () => {
    render(<MemoryRouter><Navbar /></MemoryRouter>);
    expect(screen.getByRole('navigation', { name: /main navigation/i })).toBeTruthy();
  });
});

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div>safe content</div>
      </ErrorBoundary>
    );
    expect(screen.getByText('safe content')).toBeTruthy();
  });

  it('renders fallback when child throws', () => {
    function ThrowingComp() { throw new Error('boom'); }
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary>
        <ThrowingComp />
      </ErrorBoundary>
    );
    expect(screen.getByRole('alert')).toBeTruthy();
    vi.restoreAllMocks();
  });
});
