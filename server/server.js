// server.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3001; // ✅ Dynamic port for Render

app.use(cors());
app.use(express.json());

// ✅ Quote Generator
const quotes = [
  "Believe you can and you're halfway there.",
  "Success is not final, failure is not fatal.",
  "The only limit to our realization of tomorrow is our doubts of today.",
  "Do something today that your future self will thank you for."
];

app.get('/api/quote', (req, res) => {
  const randomIndex = Math.floor(Math.random() * quotes.length);
  res.json({ quote: quotes[randomIndex] });
});

// ✅ Weather API (OpenWeatherMap)
app.get('/api/weather', async (req, res) => {
  const city = req.query.city || 'London';
  const apiKey = process.env.WEATHER_API_KEY;
  const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;

  console.log('🔍 Requesting:', url);

  try {
    const response = await axios.get(url);
    const { temp } = response.data.main;
    const description = response.data.weather[0].description;
    res.json({ city, temperature: temp, description });
  } catch (error) {
    console.error('❌ Weather API error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to fetch weather data' });
  }
});

// ✅ Currency Conversion API (Frankfurter)
app.get('/api/currency', async (req, res) => {
  const { from = 'USD', to = 'INR', amount = 1 } = req.query;
  const url = `https://api.frankfurter.app/latest?amount=${amount}&from=${from}&to=${to}`;

  try {
    const response = await axios.get(url);
    res.json(response.data);
  } catch (error) {
    console.error('❌ Currency API error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Currency conversion failed' });
  }
});

// ✅ Optional: Health check route
app.get('/api/ping', (req, res) => {
  res.send('pong');
});

// ✅ Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});