import React, { useState } from 'react';
import axios from 'axios';
import { Button, TextField, Grid, Box, Paper, Typography, CircularProgress, Alert } from '@mui/material';
import { styled } from '@mui/system';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import ResultGauge from './ResultGauge'; // A new component for displaying the result

const Input = styled('input')({
  display: 'none',
});

const initialFormState = {
  Area: '',
  Floor: '',
  Num_Bedrooms: '',
  Num_Bathrooms: '',
  Property_Age: '',
  Proximity: '',
};

const PredictionForm = () => {
  const [formData, setFormData] = useState(initialFormState);
  const [image, setImage] = useState(null);
  const [imageName, setImageName] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleImageChange = (e) => {
    if (e.target.files[0]) {
      setImage(e.target.files[0]);
      setImageName(e.target.files[0].name);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    const submissionData = new FormData();
    for (const key in formData) {
      submissionData.append(key, formData[key]);
    }
    if (image) {
      submissionData.append('image', image);
    } else {
      setError('Please upload a property image.');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post('http://localhost:8000/api/predict/', submissionData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data.investment_probability);
    } catch (err) {
      setError(err.response?.data?.error || 'An unexpected error occurred.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          {Object.keys(initialFormState).map((key) => (
            <Grid item xs={12} sm={6} key={key}>
              <TextField
                fullWidth
                required
                type="number"
                label={key.replace('_', ' ')}
                name={key}
                value={formData[key]}
                onChange={handleInputChange}
                variant="outlined"
              />
            </Grid>
          ))}
          <Grid item xs={12}>
            <label htmlFor="contained-button-file">
              <Input accept="image/*" id="contained-button-file" type="file" onChange={handleImageChange} />
              <Button fullWidth variant="outlined" component="span" startIcon={<UploadFileIcon />}>
                {imageName ? `Image: ${imageName}` : 'Upload Property Image'}
              </Button>
            </label>
          </Grid>
          <Grid item xs={12} sx={{ textAlign: 'center' }}>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              size="large"
              disabled={loading}
              sx={{ py: 1.5, px: 5, fontSize: '1.1rem' }}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Analyze Investment'}
            </Button>
          </Grid>
        </Grid>
      </form>

      {error && <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>}
      
      {result !== null && (
        <Box mt={5} textAlign="center">
            <Typography variant="h5" gutterBottom>Analysis Result</Typography>
            <ResultGauge value={result} />
        </Box>
      )}
    </Paper>
  );
};

export default PredictionForm;