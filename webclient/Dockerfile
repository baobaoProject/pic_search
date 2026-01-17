# => Build container
FROM node:alpine as builder
WORKDIR /app
COPY package.json .
RUN yarn
COPY . .
RUN NODE_OPTIONS='--openssl-legacy-provider' yarn build

# => Run container
FROM nginx:1.17-alpine

# Nginx config
RUN rm -rf /etc/nginx/conf.d
COPY conf /etc/nginx

# Static build
COPY --from=builder /app/build /usr/share/nginx/html/

# Default port exposure
EXPOSE 80

# Copy .env file and shell script to container
WORKDIR /usr/share/nginx/html
COPY ./env.sh .
COPY .env .

# Add bash
RUN apk add --no-cache bash

# Make sure env.sh has execution permissions and convert line endings to Unix format
RUN chmod +x env.sh
RUN sed -i 's/\r$//' env.sh

# Start Nginx server
CMD ["/bin/bash", "-c", "/usr/share/nginx/html/env.sh && nginx -g \"daemon off;\""]


# docker build   -t  milvusbootcamp/pic-search-webclient:2.1  .