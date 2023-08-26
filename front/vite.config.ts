import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: true,
        port: 9000,
        proxy: {
            '/answering': 'http://172.29.29.3:9000',
            '/rate': 'http://172.29.29.3:9000',
            '/upload_dataset': 'http://172.29.29.3:9000'
        }
    }
})
