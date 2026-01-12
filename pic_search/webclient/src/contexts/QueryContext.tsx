import React, { FC, createContext, ReactNode } from "react";
import axios from "axios";
import * as URL from "../utils/Endpoints";

const axiosInstance = axios.create({
  headers: {
    'Content-Type': 'application/json'
  }
});

// 添加请求拦截器，确保数据以 JSON 格式发送
axiosInstance.interceptors.request.use(
  (config: any) => {
    // 如果是 FormData，不需要进行 JSON 序列化，并且需要删除默认的 Content-Type 以便浏览器自动设置 boundary
    if (config.data instanceof FormData) {
      // 确保 headers 对象存在
      config.headers = config.headers || {};
      // 彻底删除 Content-Type，防止 Axios 默认值干扰
      delete config.headers['Content-Type'];
      // 针对某些 Axios 版本的兼容性处理
      if (config.headers.common && config.headers.common['Content-Type']) {
        delete config.headers.common['Content-Type'];
      }
      if (config.headers.post && config.headers.post['Content-Type']) {
        delete config.headers.post['Content-Type'];
      }
      return config;
    }

    // 如果是 POST/PUT/PATCH 请求且有数据，则确保数据以正确格式发送
    if (['post', 'put', 'patch'].includes(config.method || '') && config.data) {
      // 检查数据是否已经是字符串，如果不是则转换为 JSON 字符串
      if (typeof config.data !== 'string') {
        // 确保数据是正确格式的对象
        config.data = JSON.stringify(config.data);
      }
    }
    return config;
  },
  (error: any) => {
    return Promise.reject(error);
  }
);

export const queryContext = createContext<any>({});
const Provider = queryContext.Provider;

const QueryProvider: FC<{ children: ReactNode }> = ({ children }: { children: ReactNode }) => {
  const errorParser = (e: any) => {
    console.log(e);
  };
  const process = async (params: any) => {
    const url = URL.Processing;
    return await axiosInstance.get(url, { params }).catch(errorParser);
  };
  const count = async (params: any) => {
    const url = URL.Count;
    // 确保参数以正确的格式发送
    const payload = params || {};
    return await axiosInstance.post(url, payload).catch(errorParser);
  };
  const train = async (params: any) => {
    const url = URL.Train;
    return await axiosInstance.post(url, params).catch(errorParser);
  };

  const search = async (params: any) => {
    const url = URL.Search;
    return await axiosInstance.post(url, params).catch(errorParser);
  };
  const clearAll = async () => {
    const url = URL.ClearAll;
     // 确保参数以正确的格式发送
    const payload = {};
    return await axiosInstance.post(url,payload).catch(errorParser);
  };

  return (
    <Provider
      value={{
        process,
        count,
        search,
        clearAll,
        train
      }}
    >
      {children}
    </Provider>
  );
};

export default QueryProvider;