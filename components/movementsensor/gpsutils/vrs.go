// Package gpsutils implements functions that are used in the gpsrtkserial and gpsrtkpmtk.
package gpsutils

import (
	"bufio"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"sync"
	"time"

	"go.viam.com/rdk/logging"
)

const (
	vrsGGARateSec = 20 // rate to send GGA messages to the VRS in seconds
)

// VRS contains the VRS.
type VRS struct {
	ntripInfo               *NtripInfo
	readerWriter            *bufio.ReadWriter
	conn                    net.Conn
	activeBackgroundWorkers sync.WaitGroup
	cancelCtx               context.Context
	cancelFunc              func()
	logger                  logging.Logger
}

// ConnectToVirtualBase is responsible for establishing a connection to
// a virtual base station using the NTRIP protocol with enhanced error handling and retries.
func ConnectToVirtualBase(ctx context.Context, ntripInfo *NtripInfo, logger logging.Logger) (*VRS, error) {
	mp := "/" + ntripInfo.MountPoint
	credentials := ntripInfo.username + ":" + ntripInfo.password
	credentialsBase64 := base64.StdEncoding.EncodeToString([]byte(credentials))

	// Process the server URL
	serverAddr, err := url.Parse(ntripInfo.URL)
	if err != nil {
		return nil, err
	}

	conn, err := net.Dial("tcp", serverAddr.Host)
	if err != nil {
		logger.Errorf("Failed to connect to server %s: %v", serverAddr, err)

		return nil, err
	}

	rw := bufio.NewReadWriter(bufio.NewReader(conn), bufio.NewWriter(conn))
	// Construct HTTP headers with CRLF line endings
	httpHeaders := "GET " + mp + " HTTP/1.1\r\n" +
		"Host: " + serverAddr.Host + "\r\n" +
		"Authorization: Basic " + credentialsBase64 + "\r\n" +
		"Accept: */*\r\n" +
		"Ntrip-Version: Ntrip/2.0\r\n" +
		"User-Agent: NTRIP viam\r\n\r\n"

	// Send HTTP headers over the TCP connection
	_, err = rw.Write([]byte(httpHeaders))
	if err != nil {
		return nil, fmt.Errorf("failed to send HTTP headers: %w %w", err, conn.Close())
	}
	err = rw.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to write to buffer: %w  %w", err, conn.Close())
	}

	logger.Debugf("request header: %v\n", httpHeaders)
	logger.Debug("HTTP headers sent successfully.")
	cancelCtx, cancel := context.WithCancel(ctx)
	vrs := &VRS{ntripInfo: ntripInfo, readerWriter: rw, conn: conn, cancelCtx: cancelCtx, cancelFunc: cancel, logger: logger}
	return vrs, nil
}

// HasVRSStream returns the NMEA field associated with the given mountpoint
// and whether it is a Virtual Reference Station.
func HasVRSStream(sourceTable *Sourcetable, mountPoint string) (bool, error) {
	stream, isFound := sourceTable.HasStream(mountPoint)

	if !isFound {
		return false, fmt.Errorf("can not find mountpoint %s in sourcetable", mountPoint)
	}

	return stream.Nmea, nil
}

// Close closes the VRS connection and any other background threads.
func (vrs *VRS) Close() error {
	vrs.cancelFunc()
	vrs.activeBackgroundWorkers.Wait()
	return vrs.conn.Close()
}

// StartGGAThread starts a thread that writes GGA messages to the VRS.
func (vrs *VRS) StartGGAThread(ggaFunc func() (string, error)) {
	vrs.activeBackgroundWorkers.Add(1)
	go func() {
		defer vrs.activeBackgroundWorkers.Done()
		ticker := time.NewTicker(time.Duration(vrsGGARateSec) * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-vrs.cancelCtx.Done():
				return
			case <-ticker.C:
				// We currently only write the GGA message when we try to reconnect to VRS. Some documentation for VRS states that we
				// should try to send a GGA message every 5-60 seconds, but more testing is needed to determine if that is required.
				// get the GGA message from cached data
				ggaMessage, err := ggaFunc()
				if err != nil {
					vrs.logger.Error("Failed to get GGA message: ", err)
					continue
				}

				vrs.logger.Debugf("Writing GGA message: %v\n", ggaMessage)

				err = vrs.WriteLine(ggaMessage)
				if err != nil {
					vrs.logger.Error("failed to write to buffer: ", err)
					continue
				}
			}
		}
	}()
}

// ReadLine reads a line from the vrs's readerWriter.
func (vrs *VRS) ReadLine() (string, error) {
	line, _, err := vrs.readerWriter.ReadLine()
	if err != nil {
		if errors.Is(err, io.EOF) {
			vrs.readerWriter = nil
			return "", err
		}
		vrs.logger.Error("Failed to read server response: ", err)
		return "", err
	}
	return string(line), nil
}

// WriteLine writes a line to the vrs's readerWriter.
func (vrs *VRS) WriteLine(line string) error {
	_, err := vrs.readerWriter.WriteString(line)
	if err != nil {
		return err
	}

	err = vrs.readerWriter.Flush()
	if err != nil {
		return err
	}
	return nil
}

// GetReaderWriter gets the readerWriter that wraps the tcp connection for the VRS.
func (vrs *VRS) GetReaderWriter() *bufio.ReadWriter {
	return vrs.readerWriter
}
