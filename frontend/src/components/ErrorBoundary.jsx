import { Component } from 'react';

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Structured frontend error log — no sensitive payload data
    console.error('[RoomRadar:ErrorBoundary]', {
      message: error.message,
      componentStack: info.componentStack,
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div role="alert" className="alert alert-error" style={{ margin: '2rem auto', maxWidth: '600px' }}>
            <strong>Something went wrong.</strong>{' '}
            {this.state.error?.message && (
              <span style={{ display: 'block', marginTop: '.25rem', fontSize: '.875rem' }}>
                {this.state.error.message}
              </span>
            )}
            <button
              className="btn btn-sm btn-outline"
              style={{ marginTop: '.75rem' }}
              onClick={() => this.setState({ hasError: false, error: null })}
            >
              Try again
            </button>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
