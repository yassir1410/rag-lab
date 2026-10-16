import '@testing-library/jest-dom';

// Polyfill window.fetch if needed in test environment
if (!global.fetch) {
  global.fetch = () =>
    Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ status: 'ok', version: '0.2.0' }),
    });
}
