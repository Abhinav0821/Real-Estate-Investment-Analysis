import React from 'react';
import { Container, Typography, Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import PredictionForm from './components/PredictionForm';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bfa5', // A modern teal
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 300,
      letterSpacing: '0.05em',
    }
  }
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md">
        <Box textAlign="center" my={5}>
          <Typography variant="h3" component="h1" gutterBottom>
            Real Estate Investment Analyzer
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            Leveraging AI to predict high-value property investments.
          </Typography>
        </Box>
        <PredictionForm />
      </Container>
    </ThemeProvider>
  );
}

export default App;