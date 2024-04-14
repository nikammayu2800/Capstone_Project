import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5000'; // Update with your backend URL

const getComments = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/get_comments`);
        return response.data;
    } catch (error) {
        console.error('Error fetching comments:', error.message);
        throw error; // Re-throw the error to handle it elsewhere
    }
};

const getPosts = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/get_posts`);
        return response.data;
    } catch (error) {
        console.error('Error fetching posts:', error.message);
        throw error;
    }
};

export { getComments, getPosts };
