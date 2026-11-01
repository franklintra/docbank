package store

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMissingProcessingTargetsPagesCurrentLiveVersions(t *testing.T) {
	t.Parallel()
	s := newTestStore(t)
	directory, err := s.Mkdir(t.Context(), s.RootID(), "temporary")
	require.NoError(t, err)
	for i := range 3 {
		_, err := s.CreateFile(t.Context(), s.RootID(), fmt.Sprintf("text-%02d.txt", i), fakeHash("a1"), 12, "text/plain")
		require.NoError(t, err)
	}
	hidden, err := s.CreateFile(t.Context(), directory.ID, "hidden.txt", fakeHash("a1"), 12, "text/plain")
	require.NoError(t, err)
	directory, err = s.NodeByID(t.Context(), directory.ID)
	require.NoError(t, err)
	_, _, err = s.Trash(t.Context(), directory.ID, directory.Revision)
	require.NoError(t, err)
	first, err := s.MissingProcessingTargetsAfter(t.Context(), fakeHash("f1"), "", 1, 2)
	require.NoError(t, err)
	require.Len(t, first, 2)
	last, err := s.MissingProcessingTargetsAfter(t.Context(), fakeHash("f1"), first[1].ContentVersionID, 1, 2)
	require.NoError(t, err)
	require.Len(t, last, 1)
	require.Less(t, first[0].ContentVersionID, first[1].ContentVersionID)
	require.Less(t, first[1].ContentVersionID, last[0].ContentVersionID)
	for _, item := range append(first, last...) {
		require.NotEqual(t, hidden.CurrentVersionID, item.ContentVersionID)
	}
}
