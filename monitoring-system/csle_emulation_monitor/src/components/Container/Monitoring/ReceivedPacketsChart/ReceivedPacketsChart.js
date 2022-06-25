import React from 'react';
import './ReceivedPacketsChart.css';
import {
    CartesianGrid,
    Label,
    Legend, Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";

const ReceivedPacketsChart = React.memo((props) => {
        const width = 500
        const height = 200
        const margin = {
            top: 10,
            right: 30,
            left: 60,
            bottom: 25
        }

        if (props.stats !== undefined && props.stats.length > 0) {
            var minPackets = 1000000000000
            var maxPackets = 0
            const data = props.stats.map((port_stats, index) => {
                if (parseInt(port_stats.total_num_received_packets) < minPackets){
                    minPackets = parseInt(port_stats.total_num_received_packets)
                }
                if (parseInt(port_stats.total_num_received_packets) > maxPackets){
                    maxPackets = parseInt(port_stats.total_num_received_packets)
                }
                return {
                    "t": (index + 1),
                    "Received packets": parseInt(port_stats.total_num_received_packets)
                }
            })
            var domain = [0, Math.max(1, data.length)]
            var yDomain = [minPackets, maxPackets]

            return (
                <ResponsiveContainer width='100%' height={300}>
                    <LineChart
                        width={width}
                        height={height}
                        data={data}
                        margin={margin}
                    >
                        <text x={650} y={20} fill="black" textAnchor="middle" dominantBaseline="central">
                            <tspan fontSize="22">Received packets</tspan>
                        </text>
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="t" type="number" domain={domain}>
                            <Label value="Time-step t" offset={-20} position="insideBottom" className="largeFont"/>
                        </XAxis>
                        <YAxis type="number" domain={yDomain}>
                            <Label angle={270} value="# Received packets" offset={-50} position="insideLeft"
                                   className="largeFont"
                                   dy={50}/>
                        </YAxis>
                        <Tooltip/>
                        <Legend verticalAlign="top" wrapperStyle={{position: 'relative', fontSize: '22px'}}
                                className="largeFont"/>
                        <Line isAnimationActive={props.animation} animation={props.animation} type="monotone"
                              dataKey="Received packets"
                              stroke="#8884d8" addDot={false} activeDot={{r: 8}}
                              animationEasing={'linear'}
                              animationDuration={((1 - (props.animationDuration / 100)) * props.animationDurationFactor)}/>
                    </LineChart>
                </ResponsiveContainer>
            )

        } else {
            return (
                <div></div>
            )
        }
    }
)
ReceivedPacketsChart.propTypes = {};

ReceivedPacketsChart.defaultProps = {};

export default ReceivedPacketsChart;
