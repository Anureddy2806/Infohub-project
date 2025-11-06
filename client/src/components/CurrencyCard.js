import React, { useEffect, useState } from 'react';

function CurrencyCard() {
  const [rate, setRate] = useState(undefined);

  useEffect(() => {
    const base = process.env.REACT_APP_API_URL;
   
    console.log('✅ API base:', base);

    const url = `${base}/api/currency`;
    console.log('🌐 Fetching from:', url);

    fetch(url)
      .then(res => {
        console.log('📦 Response status:', res.status);
        return res.json();
      })
      .then(data => {
        console.log('📨 Currency data received:', data);
        setRate(data.rates?.INR);
      })
      .catch(err => console.error('❌ Currency fetch error:', err));
  }, []);

  return (
    <div className="module-card currency">
      <h2>Currency</h2>
      {rate !== undefined ? (
        <p>1 USD = {rate} INR</p>
      ) : (
        <p>Loading currency...</p>
      )}
    </div>
  );
}

export default CurrencyCard;