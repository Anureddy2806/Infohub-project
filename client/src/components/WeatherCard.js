import React, { useEffect, useState } from 'react';

function WeatherCard() {
  const [weather, setWeather] = useState(undefined);

  useEffect(() => {
    const base = process.env.REACT_APP_API_BASE;
    console.log('✅ API base:', base);

    const url = `${base}/api/weather`;
    console.log('🌐 Fetching from:', url);

    fetch(url)
      .then(res => {
        console.log('📦 Response status:', res.status);
        return res.json();
      })
      .then(data => {
        console.log('📨 Weather data received:', data);
        setWeather(data);
      })
      .catch(err => console.error('❌ Weather fetch error:', err));
  }, []);

  return (
    <div className="module-card weather">
      <h2>Weather</h2>
      {weather !== undefined ? (
        <p>
          {weather.city}: {weather.temperature}°C<br />
          Condition: {weather.description}
        </p>
      ) : (
        <p>Loading weather...</p>
      )}
    </div>
  );
}

export default WeatherCard;