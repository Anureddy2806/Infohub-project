import './App.css';
import WeatherCard from './components/WeatherCard';
import CurrencyCard from './components/CurrencyCard';
import QuoteCard from './components/QuoteCard';

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🌐 InfoHub</h1>
        <p>Your dashboard for weather, currency, and daily quotes.</p>
      </header>

      <main className="module-grid">
        <WeatherCard />
        <CurrencyCard />
        <QuoteCard />
      </main>

      <footer className="app-footer">
        <p>© 2025 InfoHub. Built with React.</p>
      </footer>
    </div>
  );
}

export default App;
