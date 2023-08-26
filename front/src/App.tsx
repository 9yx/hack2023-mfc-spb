import React, {useEffect, useState} from "react";
import {DislikeFilled, DislikeOutlined, InboxOutlined, LikeFilled, LikeOutlined} from "@ant-design/icons";
import {addResponseMessage, renderCustomComponent, Widget} from "react-chat-widget"
import 'react-chat-widget/lib/styles.css';
import './App.css'
import classNames from "classnames";
import {Badge, Col, ConfigProvider, Descriptions, DescriptionsProps, Layout, Row, Typography, Upload} from "antd";
import {Spb} from "./spb.tsx";

const items: DescriptionsProps['items'] = [
    {
        key: '1',
        label: 'Product',
        children: 'Cloud Database',
    },
    {
        key: '2',
        label: 'Billing Mode',
        children: 'Prepaid',
    },
    {
        key: '3',
        label: 'Automatic Renewal',
        children: 'YES',
    },
    {
        key: '4',
        label: 'Order time',
        children: '2018-04-24 18:00:00',
    },
    {
        key: '5',
        label: 'Usage Time',
        children: '2019-04-24 18:00:00',
        span: 2,
    },
    {
        key: '6',
        label: 'Status',
        children: <Badge status="processing" text="Running" />,
        span: 3,
    },
    {
        key: '7',
        label: 'Negotiated Amount',
        children: '$80.00',
    },
    {
        key: '8',
        label: 'Discount',
        children: '$20.00',
    },
    {
        key: '9',
        label: 'Official Receipts',
        children: '$60.00',
    },
    {
        key: '10',
        label: 'Config Info',
        children: (
            <>
                Data disk type: MongoDB
                <br />
                Database version: 3.4
                <br />
                Package: dds.mongo.mid
                <br />
                Storage space: 10 GB
                <br />
                Replication factor: 3
                <br />
                Region: East China 1
                <br />
            </>
        ),
    },
];

const ResponseComponent: React.FC<{
    id: string,
    answer: string
}> = (props) => {
    const [type, setType] = useState<string | undefined>();
    return (
        <div
            className={'rcw-response'}
        >
            <div className="rcw-message-text">
                {props.answer}
            </div>
            <div
                className={'mark'}
                // style={{visibility: show ? 'visible' : 'hidden'}}
            >
                <div
                    className={classNames('like', {
                        active: type === 'like'
                    })}
                    style={{display: type === 'dislike' ? 'none' : 'flex'}}
                    onClick={() => {
                        if (type === 'like') {
                            return
                        }
                        fetch('/rate', {
                            method: "POST",
                            headers: {
                                'Content-Type': 'application/json;charset=utf-8'
                            },
                            body: JSON.stringify({
                                index: props.id,
                                like: true
                            })
                        }).then(() => {
                            setType('like');
                        }).catch((e) => {
                            console.log(e)
                        })
                    }}
                >
                    {type === 'like' ? (
                        <LikeFilled rev={undefined}/>
                    ) : (
                        <LikeOutlined rev={undefined}/>
                    )}
                </div>
                <div
                    className={classNames('dislike', {
                        active: type === 'dislike'
                    })}
                    style={{display: type === 'like' ? 'none' : 'flex'}}
                    onClick={() => {
                        if (type === 'dislike') {
                            return
                        }
                        fetch('/rate', {
                            method: "POST",
                            headers: {
                                'Content-Type': 'application/json;charset=utf-8'
                            },
                            body: JSON.stringify({
                                index: props.id,
                                like: false
                            })
                        }).then(() => {
                            setType('dislike');
                        }).catch((e) => {
                            console.log(e)
                        })
                    }}
                >
                    {type === 'dislike' ? (
                        <DislikeFilled rev={undefined}/>
                    ) : (
                        <DislikeOutlined rev={undefined}/>
                    )}
                </div>
            </div>
        </div>
    )
}

function App() {
    useEffect(() => {
        addResponseMessage('Привет, я Консултина - твой персональный помощник по государственным услугам. Чем я могу тебе помочь?');
    }, []);

    return (
        <div className="App">
            <ConfigProvider
                theme={{
                    token: {
                        colorPrimary: "#E04E39"
                    }
                }}
            >
                <Layout style={{height: "100%"}}>
                    <Layout.Header className={'header-page'}>
                        <Row justify={"center"}>
                            <Col span={18}>
                                <Row justify={"space-between"}>
                                    <Col span={12}>
                                        <Typography.Title className={"title"} level={3}>
                                            Панель управления ботом
                                        </Typography.Title>
                                    </Col>
                                    <Col span={12}>
                                        <Row justify={"end"}>
                                            <Spb/>
                                        </Row>
                                    </Col>
                                </Row>
                            </Col>
                        </Row>
                    </Layout.Header>
                    <Layout.Content className={'content-page'}>
                        <Row justify={"center"}>
                            <Col span={18}>
                                <Descriptions title="Статистика" bordered items={items} />
                            </Col>
                        </Row>
                       <Row justify={"center"}>
                           <Col span={18}>
                               <Upload.Dragger
                                   action={"/upload_dataset"}
                               >
                                   <p className="ant-upload-drag-icon">
                                       <InboxOutlined rev={undefined}/>
                                   </p>
                                   <p className="ant-upload-text">Нажмите или перетащите файл в область загрузки</p>
                                   <p className="ant-upload-hint">
                                       Загрузить CSV файл для обучения модели
                                   </p>
                               </Upload.Dragger>
                           </Col>
                       </Row>
                    </Layout.Content>
                </Layout>
            </ConfigProvider>
            <Widget
                title="Консултина"
                subtitle="Ваш проводник в мире услуг"
                senderPlaceHolder={'Задайте Ваш вопрос'}
                handleNewUserMessage={async (question: string): Promise<void> => {
                    const response = await fetch('/answering', {
                        method: "POST",
                        headers: {
                            'Content-Type': 'application/json;charset=utf-8'
                        },
                        body: JSON.stringify({
                            question
                        })
                    })
                    const result: { answer: string, index: string } = await response.json();
                    console.log(result.answer)

                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-ignore
                    renderCustomComponent((props) => {
                        console.log(props)
                        return (
                            <ResponseComponent
                                id={props.index}
                                answer={props.answer}
                            />
                        )
                    }, {
                        index: result.index,
                        answer: result.answer
                    });
                }}
            />
        </div>
    )
}

export default App
