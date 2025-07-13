// Simplified example of a result display
import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

const getSentiment = (value) => {
    if (value > 0.75) return { text: "Excellent Investment", color: "success" };
    if (value > 0.5) return { text: "Good Investment", color: "info" };
    if (value > 0.25) return { text: "Consider with Caution", color: "warning" };
    return { text: "Poor Investment", color: "error" };
}

const ResultGauge = ({ value }) => {
    const probability = Math.round(value * 100);
    const sentiment = getSentiment(value);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h2" component="div" color={sentiment.color + ".main"}>
                {`${probability}%`}
            </Typography>
            <Typography variant="subtitle1" color="textSecondary" gutterBottom>
                Investment Potential
            </Typography>
            <Box sx={{ width: '80%', mt: 1 }}>
                <LinearProgress variant="determinate" value={probability} color={sentiment.color} sx={{ height: 10, borderRadius: 5 }} />
            </Box>
            <Typography variant="h6" sx={{ mt: 2 }}>
                {sentiment.text}
            </Typography>
        </Box>
    );
};

export default ResultGauge;